import gym
import numpy as np
import torch as t
from torch import nn
from torch.distributions.categorical import Categorical
import copy

device = t.device("cuda" if t.cuda.is_available() else "cpu")
Arr = np.ndarray

from PPO_utils import make_env, set_global_seeds
# import PPO as solutions

def test_get_actor_and_critic(get_actor_and_critic, mode="classic-control"):
    if mode == "atari":
        num_envs = 6
        envs = gym.vector.SyncVectorEnv([make_env("ALE/Breakout-v5", i, i, False, "", mode="atari") for i in range(num_envs)])
        num_actions = envs.single_action_space.n
        actor, critic = get_actor_and_critic(envs, mode="atari")
        actor = actor.to(device)
        critic = critic.to(device)
        obs = t.tensor(envs.reset(), device=device, dtype=t.float32)
        with t.inference_mode():
            action = actor(obs)
            value = critic(obs)
        assert action.shape == (num_envs, num_actions), f"action.shape = {action.shape}"
        assert value.shape == (num_envs, 1), f"value.shape = {value.shape}"

    elif mode == "classic-control":
        import PPO as solutions
        envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", i, i, False, "test-run") for i in range(4)])
        actor, critic = get_actor_and_critic(envs, mode="classic-control")
        actor = actor.to(device)
        critic = critic.to(device)
        actor_soln, critic_soln = solutions.get_actor_and_critic(envs)
        actor_soln = actor_soln.to(device)
        critic_soln = critic_soln.to(device)
        assert sum(p.numel() for p in actor.parameters()) == sum(p.numel() for p in actor_soln.parameters()) # 4610
        assert sum(p.numel() for p in critic.parameters()) == sum(p.numel() for p in critic_soln.parameters()) # 4545
        for name, param in actor.named_parameters():
            if "bias" in name:
                t.testing.assert_close(param.pow(2).sum().cpu(), t.tensor(0.0))
        for name, param in critic.named_parameters():
            if "bias" in name:
                t.testing.assert_close(param.pow(2).sum().cpu(), t.tensor(0.0))

    elif mode == "mujoco":
        import PPO as solutions
        envs = gym.vector.SyncVectorEnv([make_env("Hopper-v3", i, i, False, "test-run") for i in range(4)])
        actor, critic = get_actor_and_critic(envs, mode="mujoco")
        actor = actor.to(device)
        critic = critic.to(device)
        actor_soln, critic_soln = solutions.get_actor_and_critic(envs)
        actor_soln = actor_soln.to(device)
        critic_soln = critic_soln.to(device)
        assert sum(p.numel() for p in critic.parameters()) == sum(p.numel() for p in critic_soln.parameters()) # 4545
        assert sum(p.numel() for p in actor.actor_mu.parameters()) == sum(p.numel() for p in actor_soln.parameters()) # 4610
        for name, param in actor.named_parameters():
            if "bias" in name:
                t.testing.assert_close(param.pow(2).sum().cpu(), t.tensor(0.0))
        for name, param in critic.named_parameters():
            if "bias" in name:
                t.testing.assert_close(param.pow(2).sum().cpu(), t.tensor(0.0))

    print(f"All tests in `test_get_actor_and_critic(mode={mode!r})` passed!")


def test_minibatch_indexes(minibatch_indexes):
    rng = np.random.default_rng(0)
    batch_size = 16
    minibatch_size = 4
    indexes = minibatch_indexes(rng, batch_size, minibatch_size)
    assert np.array(indexes).shape == (batch_size // minibatch_size, minibatch_size)
    assert sorted(np.unique(indexes)) == list(range(batch_size))
    print("All tests in `test_minibatch_indexes` passed!")


def test_compute_advantages_single(compute_advantages, dones_false, single_env):
    import PPO as solutions
    print("".join([
        "Testing with ",
        "all dones=False" if dones_false else "episode termination",
        ", ",
        "single environment" if single_env else "multiple environments",
        " ... "
    ]))
    t_ = 5
    env_ = 1 if single_env else 12
    next_value = t.randn(env_)
    next_done = t.zeros(env_) if dones_false else t.randint(0, 2, (env_,)) 
    rewards = t.randn(t_, env_)
    values = t.randn(t_, env_)
    dones = t.zeros(t_, env_) if dones_false else t.randint(0, 2, (t_, env_))
    gamma = 0.95
    gae_lambda = 0.9
    args = (next_value, next_done, rewards, values, dones, gamma, gae_lambda)
    actual = compute_advantages(*args)
    expected = solutions.compute_advantages(*args)
    # print(actual, expected)
    t.testing.assert_close(actual, expected)

def test_compute_advantages(compute_advantages):

    for dones_false in [True, False]:
        for single_env in [True, False]:
            test_compute_advantages_single(compute_advantages, dones_false, single_env)
    print("All tests in `test_compute_advantages_single` passed!")



def test_ppo_agent(my_PPOAgent):

    import PPO as solutions
   
    args = solutions.PPOArgs(use_wandb=False, capture_video=False)
    set_global_seeds(args.seed)
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, i, i, args.capture_video, None) for i in range(4)])
    agent_solns = solutions.PPOAgent(args, envs)
    for step in range(5):
        infos_solns = agent_solns.play_step()
    
    set_global_seeds(args.seed)
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, i, i, args.capture_video, None) for i in range(4)])
    agent: solutions.PPOAgent = my_PPOAgent(args, envs)
    agent.critic = copy.deepcopy(agent_solns.critic)
    agent.actor = copy.deepcopy(agent_solns.actor)
    for step in range(5):
        infos = agent.play_step()
    
    assert agent.steps == 20, f"Agent did not take the expected number of steps: expected steps = n_steps*num_envs = 5*4 = 20, got {agent.steps}."

    actions, logprobs, values = [t.from_numpy(exp).to(device) for exp in [
        agent.memory.actions,
        agent.memory.logprobs,
        agent.memory.values,
    ]]
    actions_solns, values_solns = [t.from_numpy(exp).to(device) for exp in [
        agent_solns.memory.actions,
        agent_solns.memory.values,
    ]]

    assert (logprobs <= 0).all(), f"Agent's logprobs are not all negative."
    t.testing.assert_close(actions.cpu(), actions_solns.cpu(), msg="`actions` for agent and agent solns don't match. Make sure you're sampling actions from your actor network's logit distribution (while in inference mode).")
    t.testing.assert_close(values.cpu(), values_solns.cpu(), msg="`values` for agent and agent solns don't match. Make sure you're compute values in inference mode, by passing `self.next_obs` into the critic.")

    print("All tests in `test_agent` passed!")




def test_calc_clipped_surrogate_objective(calc_clipped_surrogate_objective):
    import PPO as solutions

    minibatch = 3
    num_actions = 4
    probs = Categorical(logits=t.randn((minibatch, num_actions)))
    mb_action = t.randint(0, num_actions, (minibatch,))
    mb_advantages = t.randn((minibatch,))
    mb_logprobs = t.randn((minibatch,))
    mb_logprobs[mb_logprobs > 0] = mb_logprobs[mb_logprobs > 0] * -1 # all logprobs should be negative
    clip_coef = 0.01
    expected = solutions.calc_clipped_surrogate_objective(probs, mb_action, mb_advantages, mb_logprobs, clip_coef)
    actual = calc_clipped_surrogate_objective(probs, mb_action, mb_advantages, mb_logprobs, clip_coef)
    t.testing.assert_close(actual.pow(2), expected.pow(2))
    if actual * expected < 0:
        print("Warning: you have calculated the negative of the policy loss, suitable for gradient descent.")
    print("All tests in `test_calc_clipped_surrogate_objective` passed.")

def test_calc_value_function_loss(calc_value_function_loss):
    import PPO as solutions

    critic = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
    mb_obs = t.randn(5, 3)
    values = critic(mb_obs)
    mb_returns = t.randn(5, 4)
    vf_coef = 0.5
    with t.inference_mode():
        expected = solutions.calc_value_function_loss(values, mb_returns, vf_coef)
        actual = calc_value_function_loss(values, mb_returns, vf_coef)
    if ((actual - expected).abs() > 1e-4) and (0.5*actual - expected).abs() < 1e-4:
        raise Exception("Your result was twice the expected value. Did you forget to use a factor of 1/2 in the mean squared difference, or the `vf_coef`?")
    t.testing.assert_close(actual, expected)
    print("All tests in `test_calc_value_function_loss` passed!")

def test_calc_entropy_bonus(calc_entropy_bonus):
    probs = Categorical(logits=t.randn((3, 4)))
    ent_coef = 0.5
    expected = ent_coef * probs.entropy().mean()
    actual = calc_entropy_bonus(probs, ent_coef)
    t.testing.assert_close(expected, actual)
    print("All tests in `test_calc_entropy_bonus` passed!")

def test_ppo_scheduler(my_PPOScheduler):
    import PPO as solutions

    args = solutions.PPOArgs()
    envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", i, i, False, "test") for i in range(4)])
    agent = solutions.PPOAgent(args, envs)

    optimizer = t.optim.Adam(agent.parameters(), lr=0.1, eps=1e-8, maximize=True)
    scheduler: solutions.PPOScheduler = my_PPOScheduler(optimizer, initial_lr=0.1, end_lr=0.5, total_training_steps=4)

    scheduler.step()
    assert (scheduler.n_step_calls == 1)
    assert abs(optimizer.param_groups[0]["lr"] - 0.2) < 1e-5
    print("All tests in `test_ppo_scheduler` passed!")

class Probe1(gym.Env):
	'''One action, observation of [0.0], one timestep long, +1 reward.

	We expect the agent to rapidly learn that the value of the constant [0.0] observation is +1.0. Note we're using a continuous observation space for consistency with CartPole.
	'''

	action_space: Discrete
	observation_space: Box

	def __init__(self):
		super().__init__()
		self.observation_space = Box(np.array([0]), np.array([0]))
		self.action_space = Discrete(1)
		self.seed()
		self.reset()

	def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
		return (np.array([0]), 1.0, True, {})

	def reset(
		self, seed: Optional[int] = None, return_info=False, options=None
	) -> Union[ObsType, Tuple[ObsType, dict]]:
		super().reset(seed=seed)
		if return_info:
			return (np.array([0.0]), {})
		return np.array([0.0])

gym.envs.registration.register(id="Probe1-v0", entry_point=Probe1)

class Probe2(gym.Env):
	'''One action, observation of [-1.0] or [+1.0], one timestep long, reward equals observation.

	We expect the agent to rapidly learn the value of each observation is equal to the observation.
	'''

	action_space: Discrete
	observation_space: Box

	def __init__(self):
		super().__init__()
		self.observation_space = Box(np.array([-1.0]), np.array([+1.0]))
		self.action_space = Discrete(1)
		self.reset()
		self.reward = None

	def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
		assert self.reward is not None
		return np.array([self.observation]), self.reward, True, {}

	def reset(
		self, seed: Optional[int] = None, return_info=False, options=None
	) -> Union[ObsType, Tuple[ObsType, dict]]:
		super().reset(seed=seed)
		self.reward = 1.0 if self.np_random.random() < 0.5 else -1.0
		self.observation = self.reward
		if return_info:
			return np.array([self.reward]), {}
		return np.array([self.reward])


gym.envs.registration.register(id="Probe2-v0", entry_point=Probe2)
	

class Probe3(gym.Env):
	'''One action, [0.0] then [1.0] observation, two timesteps, +1 reward at the end.

	We expect the agent to rapidly learn the discounted value of the initial observation.
	'''

	action_space: Discrete
	observation_space: Box

	def __init__(self):
		super().__init__()
		self.observation_space = Box(np.array([-0.0]), np.array([+1.0]))
		self.action_space = Discrete(1)
		self.reset()

	def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
		self.n += 1
		if self.n == 1:
			return np.array([1.0]), 0.0, False, {}
		elif self.n == 2:
			return np.array([0.0]), 1.0, True, {}
		raise ValueError(self.n)

	def reset(
		self, seed: Optional[int] = None, return_info=False, options=None
	) -> Union[ObsType, Tuple[ObsType, dict]]:
		super().reset(seed=seed)
		self.n = 0
		if return_info:
			return np.array([0.0]), {}
		return np.array([0.0])


gym.envs.registration.register(id="Probe3-v0", entry_point=Probe3)
	
class Probe4(gym.Env):
	'''Two actions, [0.0] observation, one timestep, reward is -1.0 or +1.0 dependent on the action.

	We expect the agent to learn to choose the +1.0 action.
	'''

	action_space: Discrete
	observation_space: Box

	def __init__(self):
		self.observation_space = Box(np.array([-0.0]), np.array([+0.0]))
		self.action_space = Discrete(2)
		self.reset()

	def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
		reward = -1.0 if action == 0 else 1.0
		return np.array([0.0]), reward, True, {}

	def reset(
		self, seed: Optional[int] = None, return_info=False, options=None
	) -> Union[ObsType, Tuple[ObsType, dict]]:
		super().reset(seed=seed)
		if return_info:
			return np.array([0.0]), {}
		return np.array([0.0])


gym.envs.registration.register(id="Probe4-v0", entry_point=Probe4)
	
class Probe5(gym.Env):
	'''Two actions, random 0/1 observation, one timestep, reward is 1 if action equals observation otherwise -1.

	We expect the agent to learn to match its action to the observation.
	'''

	action_space: Discrete
	observation_space: Box

	def __init__(self):
		self.observation_space = Box(np.array([-1.0]), np.array([+1.0]))
		self.action_space = Discrete(2)
		self.reset()

	def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
		reward = 1.0 if action == self.obs else -1.0
		return np.array([self.obs]), reward, True, {}

	def reset(
		self, seed: Optional[int] = None, return_info=False, options=None
	) -> Union[ObsType, Tuple[ObsType, dict]]:
		super().reset(seed=seed)
		self.obs = 1.0 if self.np_random.random() < 0.5 else 0.0
		if return_info:
			return np.array([self.obs], dtype=float), {}
		return np.array([self.obs], dtype=float)

def test_probe(probe_idx: int):
  '''
  Tests a probe environment by training a network on it & verifying that the value functions are
  in the expected range.
  '''
  import PPO as solutions
  # Train our network
  args = solutions.PPOArgs(
    env_id=f"Probe{probe_idx}-v0",
    exp_name=f"test-probe-{probe_idx}",
    total_timesteps=[5000, 5000, 10000, 20000, 20000][probe_idx-1],
    learning_rate=0.001,
    capture_video=False,
    use_wandb=False,
  )
  agent = solutions.train(args)

  # Get the correct set of observations, and corresponding values we expect
  obs_for_probes = [[[0.0]], [[-1.0], [+1.0]], [[0.0], [1.0]], [[0.0]], [[0.0], [1.0]]]
  expected_value_for_probes = [[[1.0]], [[-1.0], [+1.0]], [[args.gamma], [1.0]], [[1.0]], [[1.0], [1.0]]]
  expected_probs_for_probes = [None, None, None, [[0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
  tolerances = [1e-3, 1e-3, 1e-3, 2e-3, 2e-3]
  obs = t.tensor(obs_for_probes[probe_idx-1]).to(device)

  # Calculate the actual value & probs, and verify them
  with t.inference_mode():
    value = agent.critic(obs)
    probs = agent.actor(obs).softmax(-1)
  expected_value = t.tensor(expected_value_for_probes[probe_idx-1]).to(device)
  t.testing.assert_close(value, expected_value, atol=tolerances[probe_idx-1], rtol=0)
  expected_probs = expected_probs_for_probes[probe_idx-1]
  if expected_probs is not None:
    t.testing.assert_close(probs, t.tensor(expected_probs).to(device), atol=tolerances[probe_idx-1], rtol=0)
  print("Probe tests passed!\n")

def test_minibatch_indexes():
  import PPO as solutions
  rng = np.random.default_rng(0)
  batch_size = 6
  minibatch_size = 2
  indexes = solutions.minibatch_indexes(rng, batch_size, minibatch_size)

  assert np.array(indexes).shape == (batch_size // minibatch_size, minibatch_size)
  assert sorted(np.unique(indexes)) == [0, 1, 2, 3, 4, 5]
  print("All tests in `test_minibatch_indexes` passed!")

if __name__ == "__main__":
  import PPO as solutions
  test_get_actor_and_critic(solutions.get_actor_and_critic)
  test_get_actor_and_critic(solutions.get_actor_and_critic, mode="atari")
  test_compute_advantages(solutions.compute_advantages)

  probe_env = gym.make("Probe1-v0")
  assert probe_env.observation_space.shape == (1,)
  assert probe_env.action_space.shape == ()

  for probe_idx in range(1, 6):
    test_probe(probe_idx)
  test_minibatch_indexes()
  test_ppo_agent(solutions.PPOAgent)
  test_calc_clipped_surrogate_objective(solutions.calc_clipped_surrogate_objective)
  test_calc_value_function_loss(solutions.calc_value_function_loss)
  test_calc_entropy_bonus(solutions.calc_entropy_bonus)
  test_ppo_scheduler(solutions.PPOScheduler)
  