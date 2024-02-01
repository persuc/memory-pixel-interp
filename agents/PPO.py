import time
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from numpy.random import Generator
import torch as t
from torch import Tensor
from torch.optim.optimizer import Optimizer
import gym
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import einops
from typing import List, Tuple, Literal, Union, Optional
from jaxtyping import Float, Int
import wandb

from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))
import main

from PPO_utils import PPOArgs, make_env, set_global_seeds, wrap_atari_memory_env, wrap_atari_pixels_env

Arr = np.ndarray

device = t.device("cuda" if t.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module='gym.*')
warnings.filterwarnings("ignore", category=UserWarning, module='gym.*')


# 1️⃣ SETTING UP OUR AGENT

def layer_init(layer: nn.Module, std=np.sqrt(2), bias_const=0.0):
	t.nn.init.orthogonal_(layer.weight, std)
	t.nn.init.constant_(layer.bias, bias_const)
	return layer

def get_actor_and_critic(
	envs: gym.vector.SyncVectorEnv,
	mode: Literal["classic_control", "convolutional", "sparse"],
) -> Tuple[nn.Sequential, nn.Sequential]:
	'''
	Returns (actor, critic), the networks used for PPO.
	'''
	obs_shape = envs.single_observation_space.shape
	num_obs = np.array(obs_shape).prod()
	num_actions = (
		envs.single_action_space.n 
		if isinstance(envs.single_action_space, gym.spaces.Discrete) 
		else envs.single_action_space.shape[0]
	)

	match mode:
		case "classic_control":
			critic = nn.Sequential(
				nn.Flatten(),
				layer_init(nn.Linear(num_obs, 64)),
				nn.Tanh(),
				layer_init(nn.Linear(64, 64)),
				nn.Tanh(),
				layer_init(nn.Linear(64, 1), std=1.0)
			)

			actor = nn.Sequential(
				nn.Flatten(),
				layer_init(nn.Linear(num_obs, 64)),
				nn.Tanh(),
				layer_init(nn.Linear(64, 64)),
				nn.Tanh(),
				layer_init(nn.Linear(64, num_actions), std=0.01)
			)
		case "convolutional":

			# for atari RAM, obs is (4, 32, 128)
	
			assert obs_shape[-1] % 8 == 4, f"Expected obs.shape[-1] ≡ 4 % 8. Got shape: {obs_shape} ({obs_shape[-1]} % 8 = {obs_shape[-1] % 8})"

			L_after_convolutions = (obs_shape[-1] // 8) - 3
			in_features = 64 * L_after_convolutions * L_after_convolutions

			hidden = nn.Sequential(
				layer_init(nn.Conv2d(4, 32, 8, stride=4, padding=0)),
				nn.ReLU(),
				layer_init(nn.Conv2d(32, 64, 4, stride=2, padding=0)),
				nn.ReLU(),
				layer_init(nn.Conv2d(64, 64, 3, stride=1, padding=0)),
				nn.ReLU(),
				nn.Flatten(),
				layer_init(nn.Linear(in_features, 512)),
				nn.ReLU(),
			)
			actor = nn.Sequential(
				hidden,
				layer_init(nn.Linear(512, num_actions), std=0.01)
			)
			critic = nn.Sequential(
				hidden,
				layer_init(nn.Linear(512, 1), std=1)
			)
	
		case "sparse":
			raise NotImplementedError("See `mujoco.py`.")
		
		case _:
			raise NotImplementedError(f"Mode not recognised: {mode}")

	return actor.to(device), critic.to(device)


def shift_rows(arr: Tensor):
  """
  Helper function for compute_advantages_vectorized

  Given a 1D array like:
    [1, 2, 3]
  this function will return:
    [[1, 2, 3],
    [0, 1, 2],
    [0, 0, 1]]

  If the array has >1D, it treats the later dimensions as batch dims (i.e.
  it does exactly the same thing as above, but along each slice of the
  latter dimensions). We'll use this function on arrays of shape (T, envs),
  so envs are treated as batch dims.
  """
  L = arr.shape[0]
  output = t.zeros(L, 2*L, *arr.shape[1:], device=arr.device).to(dtype=arr.dtype)
  output[:, :L] = arr[None, :]
  output = einops.rearrange(output, "i j ... -> (i j) ...")
  output = output[:L*(2*L-1)]
  output = einops.rearrange(output, "(i j) ... -> i j ...", i=L)
  output = output[:, :L]
  return output


@t.inference_mode()
def compute_advantages(
	 next_value: t.Tensor,
	 next_done: t.Tensor,
	 rewards: t.Tensor,
	 values: t.Tensor,
	 dones: t.Tensor,
	 gamma: float,
	 gae_lambda: float,
) -> t.Tensor:
  """
  Basic idea (assuming num_envs=1 in this description, but the case generalises):

    create a matrix of discount factors (gamma*gae_lambda)**l, shape (t, l), suitably shifted
    create a matrix of deltas, shape (t, l), suitably shifted
    mask the deltas after the "done" points
    multiply two matrices and sum over l (second dim)
  """
  T, num_envs = rewards.shape
  next_values = t.concat([values[1:], next_value.unsqueeze(0)])
  next_dones = t.concat([dones[1:], next_done.unsqueeze(0)])
  deltas = rewards + gamma * next_values * (1.0 - next_dones) - values

  deltas_repeated = einops.repeat(deltas, "j env -> i j env", i=T)
  mask = einops.repeat(next_dones, "j env -> i j env", i=T)
  mask_uppertri = einops.repeat(t.triu(t.ones(T, T, device=mask.device)), "i j -> i j env", env=num_envs)
  mask = mask * mask_uppertri
  mask = 1 - (mask.cumsum(dim=1) > 0).float()
  mask = t.concat([t.ones(T, 1, num_envs, device=mask.device), mask[:, :-1]], dim=1)
  mask = mask * mask_uppertri
  deltas_masked = mask * deltas_repeated

  discount_factors = (gamma * gae_lambda) ** t.arange(T, device=mask.device)
  discount_factors_repeated = einops.repeat(discount_factors, "t -> t env", env=num_envs)
  discount_factors_shifted = shift_rows(discount_factors_repeated)

  advantages = (discount_factors_shifted * deltas_masked).sum(dim=1)
  return advantages


def minibatch_indexes(rng: Generator, batch_size: int, minibatch_size: int) -> List[np.ndarray]:
	'''
	Return a list of length num_minibatches = (batch_size // minibatch_size), where each element is an
	array of indexes into the batch.

	Each index should appear exactly once.
	'''
	assert batch_size % minibatch_size == 0
	indices = rng.permutation(batch_size)
	indices = einops.rearrange(indices, "(mb_num mb_size) -> mb_num mb_size", mb_size=minibatch_size)
	return list(indices)

def to_numpy(arr: Union[np.ndarray, Tensor]):
	'''
	Converts a (possibly cuda and non-detached) tensor to numpy array.
	'''
	if isinstance(arr, Tensor):
		arr = arr.detach().cpu().numpy()
	return arr

@dataclass
class ReplayMinibatch:
	'''
	Samples from the replay memory, converted to PyTorch for use in neural network training.

	Data is equivalent to (s_t, a_t, logpi(a_t|s_t), V(s_t), A_t, ret_t, d_{t+1})
	'''	
	observations: Tensor # shape [minibatch_size, *observation_shape]
	actions: Tensor # shape [minibatch_size,]
	logprobs: Tensor # shape [minibatch_size,]
	values: Tensor # shape [minibatch_size,]
	advantages: Tensor # shape [minibatch_size,]
	returns: Tensor # shape [minibatch_size,]
	dones: Tensor # shape [minibatch_size,]

class ReplayMemory:
	'''
	Contains buffer; has a method to sample from it to return a ReplayMinibatch object.
	'''
	rng: Generator
	observations: np.ndarray # shape [buffer_size, num_envs, *observation_shape]
	actions: np.ndarray # shape [buffer_size, num_envs]
	logprobs: np.ndarray # shape [buffer_size, num_envs]
	values: np.ndarray # shape [buffer_size, num_envs]
	rewards: np.ndarray # shape [buffer_size, num_envs]
	dones: np.ndarray # shape [buffer_size, num_envs]

	def __init__(self, args: PPOArgs, envs: gym.vector.SyncVectorEnv):
		self.args = args
		self.rng = np.random.default_rng(args.seed)
		self.num_envs = envs.num_envs
		self.obs_shape = envs.single_observation_space.shape
		self.action_shape = envs.single_action_space.shape
		self.reset_memory()


	def reset_memory(self):
		'''
		Resets all stored experiences, ready for new ones to be added to memory.
		'''
		self.observations = np.empty((0, self.num_envs, *self.obs_shape), dtype=np.float32)
		self.actions = np.empty((0, self.num_envs, *self.action_shape), dtype=np.int32)
		self.logprobs = np.empty((0, self.num_envs), dtype=np.float32)
		self.values = np.empty((0, self.num_envs), dtype=np.float32)
		self.rewards = np.empty((0, self.num_envs), dtype=np.float32)
		self.dones = np.empty((0, self.num_envs), dtype=bool)


	def add(self, obs, actions, logprobs, values, rewards, dones) -> None:
		'''
		Each argument can be a PyTorch tensor or NumPy array.

		obs: shape (num_environments, *observation_shape)
			Observation before the action
		actions: shape (num_environments,)
			Action chosen by the agent
		logprobs: shape (num_environments,)
			Log probability of the action that was taken (according to old policy)
		values: shape (num_environments,)
			Values, estimated by the critic (according to old policy)
		rewards: shape (num_environments,)
			Reward after the action
		dones: shape (num_environments,)
			If True, the episode ended and was reset automatically
		'''
		assert obs.shape == (self.num_envs, *self.obs_shape)
		assert actions.shape == (self.num_envs, *self.action_shape)
		assert logprobs.shape == (self.num_envs,)
		assert values.shape == (self.num_envs,)
		assert dones.shape == (self.num_envs,)
		assert rewards.shape == (self.num_envs,)
  
		self.observations = np.concatenate((self.observations, to_numpy(obs[None, :])))
		self.actions = np.concatenate((self.actions, to_numpy(actions[None, :])))
		self.logprobs = np.concatenate((self.logprobs, to_numpy(logprobs[None, :])))
		self.values = np.concatenate((self.values, to_numpy(values[None, :])))
		self.rewards = np.concatenate((self.rewards, to_numpy(rewards[None, :])))
		self.dones = np.concatenate((self.dones, to_numpy(dones[None, :])))


	def get_minibatches(self, next_value: t.Tensor, next_done: t.Tensor) -> List[ReplayMinibatch]:
		minibatches = []

		# Stack all experiences, and move them to our device
		obs, actions, logprobs, values, rewards, dones = [t.from_numpy(exp).to(device) for exp in [
			self.observations, self.actions, self.logprobs, self.values, self.rewards, self.dones
		]]

		# Compute advantages and returns (then get the list of tensors, in the right order to add to our ReplayMinibatch)
		advantages = compute_advantages(next_value, next_done, rewards, values, dones.float(), self.args.gamma, self.args.gae_lambda)
		returns = advantages + values
		replaymemory_data = [obs, actions, logprobs, values, advantages, returns, dones]

		# Generate `batches_per_learning_phase` sets of minibatches (each set of minibatches is a shuffled permutation of
		# all the experiences stored in memory)
		for _ in range(self.args.batches_per_learning_phase):

			indices_for_each_minibatch = minibatch_indexes(self.rng, self.args.batch_size, self.args.minibatch_size)

			for indices_for_minibatch in indices_for_each_minibatch:
				minibatches.append(ReplayMinibatch(*[
					arg.flatten(0, 1)[indices_for_minibatch] for arg in replaymemory_data
				]))
	
		# Reset memory, since we only run this once per learning phase
		self.reset_memory()

		return minibatches

class PPOAgent(nn.Module):
	critic: nn.Sequential
	actor: nn.Sequential

	def __init__(self, args: PPOArgs, envs: gym.vector.SyncVectorEnv):
		super().__init__()
		self.args = args
		self.envs = envs

		# Keep track of global number of steps taken by agent
		self.steps = 0

		# Get actor and critic networks
		self.actor, self.critic = get_actor_and_critic(envs, mode=args.model_type)

		# Define our first (obs, done), so we can start adding experiences to our replay memory
		self.next_obs = t.tensor(envs.reset()).to(device, dtype=t.float)
		self.next_done = t.zeros(envs.num_envs).to(device, dtype=t.float)

		# Create our replay memory
		self.memory = ReplayMemory(args, envs)


	def play_step(self) -> List[dict]:
		'''
		Carries out a single interaction step between the agent and the environment, and adds results to the replay memory.
		'''
		# Get newest observations
		obs = self.next_obs
		dones = self.next_done
		# Compute logits based on newest observation, and use it to get an action distribution we sample from
		with t.inference_mode():
			logits = self.actor(obs)
		probs = Categorical(logits=logits)
		actions = probs.sample()
		# Step environment based on the sampled action
		next_obs, rewards, next_dones, infos = self.envs.step(actions.cpu().numpy())

		# Calculate logprobs and values, and add this all to replay memory
		logprobs = probs.log_prob(actions)
		with t.inference_mode():
			values = self.critic(obs).flatten()
		self.memory.add(obs, actions, logprobs, values, rewards, dones)

		# Set next observation, and increment global step counter
		self.next_obs = t.from_numpy(next_obs).to(device, dtype=t.float)
		self.next_done = t.from_numpy(next_dones).to(device, dtype=t.float)
		self.steps += self.envs.num_envs

		# Return infos dict, for logging
		return infos

	def get_minibatches(self) -> list[ReplayMinibatch]:
		'''
		Gets minibatches from the replay memory.
		'''
		with t.inference_mode():
			next_value = self.critic(self.next_obs).flatten()
		return self.memory.get_minibatches(next_value, self.next_done)

# 2️⃣ LEARNING PHASE

def calc_clipped_surrogate_objective(
	probs: Categorical, 
	mb_action: Int[Tensor, "minibatch_size"], 
	mb_advantages: Float[Tensor, "minibatch_size"], 
	mb_logprobs: Float[Tensor, "minibatch_size"], 
	clip_coef: float, 
	eps: float = 1e-8
) -> Float[Tensor, ""]:
	'''Return the clipped surrogate objective, suitable for maximisation with gradient ascent.

	probs:
		a distribution containing the actor's unnormalized logits of shape (minibatch_size, num_actions)
	mb_action:
		what actions actions were taken in the sampled minibatch
	mb_advantages:
		advantages calculated from the sampled minibatch
	mb_logprobs:
		logprobs of the actions taken in the sampled minibatch (according to the old policy)
	clip_coef:
		amount of clipping, denoted by epsilon in Eq 7.
	eps:
		used to add to std dev of mb_advantages when normalizing (to avoid dividing by zero)
	'''
	assert mb_action.shape == mb_advantages.shape == mb_logprobs.shape	
	logits_diff = probs.log_prob(mb_action) - mb_logprobs

	r_theta = t.exp(logits_diff)

	mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + eps)

	non_clipped = r_theta * mb_advantages
	clipped = t.clip(r_theta, 1-clip_coef, 1+clip_coef) * mb_advantages

	return t.minimum(non_clipped, clipped).mean()

def calc_value_function_loss(
	values: Float[Tensor, "minibatch_size"],
	mb_returns: Float[Tensor, "minibatch_size"],
	vf_coef: float
) -> Float[Tensor, ""]:
	'''Compute the value function portion of the loss function.

	values:
		the value function predictions for the sampled minibatch (using the updated critic network)
	mb_returns:
		the target for our updated critic network (computed as `advantages + values` from the old network)
	vf_coef:
		the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
	'''
	assert values.shape == mb_returns.shape
	return 0.5 * vf_coef * (values - mb_returns).pow(2).mean()

def calc_entropy_bonus(probs: Categorical, ent_coef: float):
	'''Return the entropy bonus term, suitable for gradient ascent.

	probs:
		the probability distribution for the current policy
	ent_coef: 
		the coefficient for the entropy loss, which weights its contribution to the overall objective function. Denoted by c_2 in the paper.
	'''
	return ent_coef * probs.entropy().mean()



class PPOScheduler:
	def __init__(self, optimizer: Optimizer, initial_lr: float, end_lr: float, total_training_steps: int):
		self.optimizer = optimizer
		self.initial_lr = initial_lr
		self.end_lr = end_lr
		self.total_training_steps = total_training_steps
		self.n_step_calls = 0

	def step(self):
		'''Implement linear learning rate decay so that after total_training_steps calls to step, the learning rate is end_lr.
		'''
		self.n_step_calls += 1
		frac = self.n_step_calls / self.total_training_steps
		assert frac <= 1
		for param_group in self.optimizer.param_groups:
			param_group["lr"] = self.initial_lr + frac * (self.end_lr - self.initial_lr)


def make_optimizer(agent: PPOAgent, total_training_steps: int, initial_lr: float, end_lr: float) -> Tuple[optim.Adam, PPOScheduler]:
	'''Return an appropriately configured Adam with its attached scheduler.'''
	optimizer = optim.Adam(agent.parameters(), lr=initial_lr, eps=1e-5, maximize=True)
	scheduler = PPOScheduler(optimizer, initial_lr, end_lr, total_training_steps)
	return (optimizer, scheduler)

# 3️⃣ TRAINING LOOP

class PPOTrainer:
	max_reward_earned: Optional[int]

	def __init__(self, args: PPOArgs, agent: Optional[PPOAgent] = None):
		"""
			Agent will be created if None. If not none, training will resume with agent as is.
		"""
		set_global_seeds(args.seed)
		self.args = args
		self.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
		made_envs = [make_env(args, args.seed + i, i, self.run_name) for i in range(args.num_envs)]
		self.envs = gym.vector.SyncVectorEnv(made_envs)
		if agent is None:
			self.agent = PPOAgent(args, self.envs).to(device)
		else:
			self.agent = agent.to(device)
			reference_agent = PPOAgent(args, self.envs).to(device)
			param_shapes = zip(self.agent.parameters(), reference_agent.parameters())
			for agent_param, reference_param in param_shapes:
				if agent_param.shape != reference_param.shape:
					print(f"Provided agent has incorrect shape.\nGot:{list(agent.parameters())}\nExpected:{list(reference_agent.parameters())}")
		self.optimizer, self.scheduler = make_optimizer(self.agent, args.total_training_steps, args.learning_rate, 0.0)
		self.epoch = 0
		self.max_reward_earned = None # TODO: implement max reward seen thus far
		if args.wandb_project_name is not None: wandb.init(
			project=args.wandb_project_name,
			entity=args.wandb_entity,
			name=self.run_name,
			monitor_gym=args.episodes_per_video is not None
		)

	def rollout_phase(self) -> Optional[float]:
		'''
		This function populates the memory with a new set of experiences, using `self.agent.play_step`
		to step through the environment. It also returns the episode length of the most recently terminated
		episode (used in the progress bar readout).
		'''
		episode_lengths = []

		self.envs.seed(self.epoch)

		for _step in range(self.args.num_steps):
			infos = self.agent.play_step()
			for info in infos:
				if "episode" in info.keys():
					episode_len = info["episode"]["l"]
					episode_return = info["episode"]["r"]
					if self.args.wandb_project_name is not None: wandb.log({
						"episode_length": episode_len,
						"episode_return": episode_return,
					}, step=self.agent.steps)
					episode_lengths.append(episode_len)
		
		# TODO: debug framy video where potentially env.step() is called more often than env.render()
		# runners: list[PythonMemoryRunner] = [e.unwrapped.runner for e in self.agent.envs.envs]
		# renders = [runner.renders for runner in runners]
		# steps = [runner.steps for runner in runners]
		# zipped = zip(renders, steps)
		# if not all([z[0] == z[1] for z in zipped]):
		# 	print(zipped)
		# 	raise Exception(f"Runner did not run steps equal to number of renders\n {zipped}")

		return (sum(episode_lengths) / len(episode_lengths)) if episode_lengths else None

	def learning_phase(self) -> None:
		'''
		This function does the following:

			- Generates minibatches from memory
			- Calculates the objective function, and takes an optimization step based on it
			- Clips the gradients (see detail #11)
			- Steps the learning rate scheduler
		'''
		minibatches = self.agent.get_minibatches()
		for minibatch in minibatches:
			objective_fn = self.compute_ppo_objective(minibatch)
			objective_fn.backward()
			nn.utils.clip_grad.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
			self.optimizer.step()
			self.optimizer.zero_grad()
			self.scheduler.step()

	def compute_ppo_objective(self, minibatch: ReplayMinibatch) -> Float[Tensor, ""]:
		'''
		Handles learning phase for a single minibatch. Returns objective function to be maximized.
		'''
		logits = self.agent.actor(minibatch.observations)
		probs = Categorical(logits=logits)
		values = self.agent.critic(minibatch.observations).squeeze()

		clipped_surrogate_objective = calc_clipped_surrogate_objective(probs, minibatch.actions, minibatch.advantages, minibatch.logprobs, self.args.clip_coef)
		value_loss = calc_value_function_loss(values, minibatch.returns, self.args.vf_coef)
		entropy_bonus = calc_entropy_bonus(probs, self.args.ent_coef)

		total_objective_function = clipped_surrogate_objective - value_loss + entropy_bonus

		with t.inference_mode():
			newlogprob = probs.log_prob(minibatch.actions)
			logratio = newlogprob - minibatch.logprobs
			ratio = logratio.exp()
			approx_kl = (ratio - 1 - logratio).mean().item()
			clipfracs = [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]
		if self.args.wandb_project_name is not None: wandb.log(dict(
			total_steps = self.agent.steps,
			values = values.mean().item(),
			learning_rate = self.scheduler.optimizer.param_groups[0]["lr"],
			value_loss = value_loss.item(),
			clipped_surrogate_objective = clipped_surrogate_objective.item(),
			entropy = entropy_bonus.item(),
			approx_kl = approx_kl,
			clipfrac = np.mean(clipfracs)
		), step=self.agent.steps)

		return total_objective_function
	
	def train(self) -> PPOAgent:
		'''Implements training loop, used like: agent = train(args)'''

		progress_bar = tqdm(range(self.args.total_phases))

		for epoch in progress_bar:
			self.epoch += 1

			avg_episode_len = self.rollout_phase()
			progress_bar.set_description(f"Epoch {epoch:02}" + f", Avg Episode Length: {avg_episode_len:.2f}" if avg_episode_len is not None else "")

			self.learning_phase()

			if self.args.save_nth_epoch is not None and self.epoch % self.args.save_nth_epoch[0] == 0:
				t.save({
						"epoch": epoch,
						"model_state_dict": self.agent.state_dict(),
						"optimizer_state_dict": self.optimizer.state_dict(),
				}, f"{self.args.save_nth_epoch[1]}-Epoch{self.epoch}.pt")
		
		self.envs.close()
		if self.args.wandb_project_name is not None:
			wandb.finish()

		return self.agent

def train_gym_memory():
	name = "PPOMemory"
	args = PPOArgs(
		env_id = "ALE/Breakout-v5",
		exp_name = name,
		model_type="convolutional",
		# wandb_project_name = name,
		clip_coef = 0.1,
		num_envs = 8,
		episodes_per_video=20,
		make_kwargs={"obs_type": "ram"},
		save_nth_epoch=(50, name),
		wrap_env=wrap_atari_memory_env,
	)

	trainer = PPOTrainer(args)
	return trainer.train()

def train_gym_pixels():
	name = "PPOPixels"
	args = PPOArgs(
		env_id = "ALE/Breakout-v5",
		model_type="convolutional",
		exp_name = name,
		wandb_project_name = name,
		clip_coef = 0.1,
		num_envs = 8,
		episodes_per_video=20,
		make_kwargs={"obs_type": "grayscale"},
		save_nth_epoch=(50, name),
		wrap_env=wrap_atari_pixels_env,
	)

	trainer = PPOTrainer(args)
	return trainer.train()

if __name__ == "__main__":
	# train_memory()
	# train_pixels()
	train_gym_memory()
	# train_gym_pixels()