# %%
from enum import Enum
import gym
import numpy as np
import random
import torch as t
from typing import Optional, Literal
from dataclasses import dataclass
import pandas as pd
from IPython.display import display
Arr = np.ndarray
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from einops import rearrange
from typing import List, Optional
from distutils.util import strtobool
import argparse

from atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv, # only if "FIRE" is in env.unwrapped.get_action_meanings()
    ClipRewardEnv,
)
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gym.wrappers.record_video import RecordVideo
from gym.wrappers.clip_action import ClipAction
from gym.wrappers.normalize import NormalizeObservation
from gym.wrappers.transform_observation import TransformObservation
from gym.wrappers.normalize import NormalizeReward
from gym.wrappers.transform_reward import TransformReward

class ModelType(Enum):
  CLASSIC_CONTROL = 1
  CONVOLUTIONAL = 2
  SPARSE = 3
  CLASSIC_CONTROL_WRAPPED = 4

def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str, mode: ModelType, video_log_freq: Optional[int] = None):
    """Return a function that returns an environment after setting up boilerplate."""

    if video_log_freq is None:
        video_log_freq = {
            ModelType.CLASSIC_CONTROL: 25,
            ModelType.CLASSIC_CONTROL_WRAPPED: 20,
            ModelType.CONVOLUTIONAL: 30,
            ModelType.SPARSE: 50
        }[mode]

    def thunk():
        env = gym.make(env_id)
        env = RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = RecordVideo(
                    env, 
                    f"videos/{run_name}", 
                    episode_trigger = lambda x : x % video_log_freq == 0
                )

        match mode:
            case ModelType.CONVOLUTIONAL:
                env = prepare_atari_env(env)
            case ModelType.SPARSE:
                env = prepare_mujoco_env(env)
            case ModelType.CLASSIC_CONTROL_WRAPPED:
                env = prepare_memory_env(env)
        
        obs = env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    
    return thunk

def prepare_memory_env(env: gym.Env):
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env, lambda env: env.unwrapped.lives())
    env = ClipRewardEnv(env)
    # TODO: fix framestack for MultiBinary obs space
    # env = FrameStack(env, num_stack=4)
    return env


def prepare_atari_env(env: gym.Env):
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env, lambda env: env.unwrapped.ale.lives())
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = ResizeObservation(env, shape=(84, 84))
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4)
    return env


def prepare_mujoco_env(env: gym.Env):
    env = ClipAction(env)
    env = NormalizeObservation(env)
    env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = NormalizeReward(env)
    env = TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)


def window_avg(arr: Arr, window: int):
    """
    Computes sliding window average
    """
    return np.convolve(arr, np.ones(window), mode="valid") / window

def cummean(arr: Arr):
    """
    Computes the cumulative mean
    """
    return np.cumsum(arr) / np.arange(1, len(arr) + 1)

# Taken from https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
# See https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
def ewma(arr : Arr, alpha : float):
    '''
    Returns the exponentially weighted moving average of x.
    Parameters:
    -----------
    x : array-like
    alpha : float {0 <= alpha <= 1}
    Returns:
    --------
    ewma: numpy array
          the exponentially weighted moving average
    '''
    # Coerce x to an array
    s = np.zeros_like(arr)
    s[0] = arr[0]
    for i in range(1,len(arr)):
        s[i] = alpha * arr[i] + (1-alpha)*s[i-1]
    return s


def sum_rewards(rewards : List[int], gamma : float = 1):
    """
    Computes the total discounted sum of rewards for an episode.
    By default, assume no discount
    Input:
        rewards [r1, r2, r3, ...] The rewards obtained during an episode
        gamma: Discount factor
    Output:
        The sum of discounted rewards 
        r1 + gamma*r2 + gamma^2 r3 + ...
    """
    total_reward = 0
    for r in rewards[:0:-1]: #reverse, excluding first
        total_reward += r
        total_reward *= gamma
    total_reward += rewards[0]
    return total_reward

@dataclass
class PPOArgs:
	exp_name: str = "PPO_Implementation"
	seed: int = 1
	cuda: bool = t.cuda.is_available()
	log_dir: str = "logs"
	use_wandb: bool = False
	wandb_project_name: str = "PPOCart"
	wandb_entity: str = None
	capture_video: bool = True
	env_id: str = "CartPole-v1"
	total_timesteps: int = 500000
	learning_rate: float = 0.00025
	num_envs: int = 4
	num_steps: int = 128
	gamma: float = 0.99
	gae_lambda: float = 0.95
	num_minibatches: int = 4
	batches_per_learning_phase: int = 4
	clip_coef: float = 0.2
	ent_coef: float = 0.01
	vf_coef: float = 0.5
	max_grad_norm: float = 0.5
	mode: Literal["classic-control", "atari", "mujoco"] = "classic-control"

	def __post_init__(self):
		self.batch_size = self.num_steps * self.num_envs
		assert self.batch_size % self.num_minibatches == 0, "batch_size must be divisible by num_minibatches"
		self.minibatch_size = self.batch_size // self.num_minibatches
		self.total_phases = self.total_timesteps // self.batch_size
		self.total_training_steps = self.total_phases * self.batches_per_learning_phase * self.num_minibatches

arg_help_strings = dict(
    exp_name = "the name of this experiment",
    seed = "seed of the experiment",
    cuda = "if toggled, cuda will be enabled by default",
    log_dir = "the directory where the logs will be stored",
    use_wandb = "if toggled, this experiment will be tracked with Weights and Biases",
    wandb_project_name = "the wandb's project name",
    wandb_entity = "the entity (team) of wandb's project",
    capture_video = "whether to capture videos of the agent performances (check out `videos` folder)",
    env_id = "the id of the environment",
    total_timesteps = "total timesteps of the experiments",
    learning_rate = "the learning rate of the optimizer",
    num_envs = "number of synchronized vector environments in our `envs` object (this is N in the '37 Implementational Details' post)",
    num_steps = "number of steps taken in the rollout phase (this is M in the '37 Implementational Details' post)",
    gamma = "the discount factor gamma",
    gae_lambda = "the discount factor used in our GAE estimation",
    num_minibatches = "the number of minibatches you divide each batch up into",
    batches_per_learning_phase = "how many times you loop through the data generated in each rollout phase",
    clip_coef = "the epsilon term used in the clipped surrogate objective function",
    ent_coef = "coefficient of entropy bonus term",
    vf_coef = "cofficient of value loss function",
    max_grad_norm = "value used in gradient clipping",
    mode = "can be 'classic-control', 'atari' or 'mujoco'",
    batch_size = "N * M in the '37 Implementational Details' post (calculated from other values in PPOArgs)",
    minibatch_size = "the size of a single minibatch we perform a gradient step on (calculated from other values in PPOArgs)",
    total_phases = "total number of phases during training (calculated from other values in PPOArgs)",
    total_training_steps = "total number of minibatches we will perform an update step on during training (calculated from other values in PPOArgs)",
)

def arg_help_PPO(args: Optional[PPOArgs], print_df=False):
    """Prints out a nicely displayed list of arguments, their default values, and what they mean."""
    if args is None:
        args = PPOArgs()
        changed_args = []
    else:
        default_args = PPOArgs()
        changed_args = [key for key in default_args.__dict__ if getattr(default_args, key) != getattr(args, key)]
    df = pd.DataFrame([arg_help_strings]).T
    df.columns = ["description"]
    df["default value"] = [repr(getattr(args, name)) for name in df.index]
    df.index.name = "arg"
    df = df[["default value", "description"]]
    if print_df:
        df.insert(1, "changed?", ["yes" if i in changed_args else "" for i in df.index])
        with pd.option_context(
            'max_colwidth', 0, 
            'display.width', 150, 
            'display.colheader_justify', 'left'
        ):
            print(df)
    else:
        s = (
            df.style
            .set_table_styles([
                {'selector': 'td', 'props': 'text-align: left;'},
                {'selector': 'th', 'props': 'text-align: left;'}
            ])
            .apply(lambda row: ['background-color: red' if row.name in changed_args else None] + [None,] * (len(row) - 1), axis=1)
        )
        with pd.option_context("max_colwidth", 0):
            display(s)

# %%

def plot_cartpole_obs_and_dones(obs: t.Tensor, done: t.Tensor):
    """
    obs: shape (n_steps, n_envs, n_obs)
    dones: shape (n_steps, n_envs)

    Plots the observations and the dones.
    """
    obs = rearrange(obs, "step env ... -> (env step) ...").cpu().numpy()
    done = rearrange(done, "step env -> (env step)").cpu().numpy()
    done_indices = np.nonzero(done)[0]
    fig = make_subplots(rows=2, cols=1, subplot_titles=["Cart x-position", "Pole angle"])
    fig.update_layout(template="simple_white", title="CartPole experiences (dotted lines = termination)", showlegend=False)
    d = dict(zip(['posn', 'speed', 'angle', 'angular_velocity'], obs.T))
    d["posn_min"] = np.full_like(d["posn"], -2.4)
    d["posn_max"] = np.full_like(d["posn"], +2.4)
    d["angle_min"] = np.full_like(d["posn"], -0.2095)
    d["angle_max"] = np.full_like(d["posn"], +0.2095)
    for i, (name0, color, y) in enumerate(zip(["posn", "angle"], px.colors.qualitative.D3, [2.4, 0.2095]), 1):
        for name1 in ["", "_min", "_max"]:
            fig.add_trace(go.Scatter(y=d[name0+name1], name=name0+name1, mode="lines", marker_color=color), col=1, row=i)
        for x in done_indices:
            fig.add_vline(x=x, y1=1, y0=0, line_width=2, line_color="black", line_dash="dash", col=1, row=i)
    for sign, text0 in zip([-1, 1], ["Min", "Max"]):
        for row, (y, text1) in enumerate(zip([2.4, 0.2095], ["posn", "angle"]), 1):
            fig.add_annotation(text=" ".join([text0, text1]), xref="paper", yref="paper", x=550, y=sign*y, showarrow=False, row=row, col=1)
    fig.show()

def set_global_seeds(seed):
    '''Sets random seeds in several different ways (to guarantee reproducibility)
    '''
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    t.backends.cudnn.deterministic = True

toggles = ["torch_deterministic", "cuda", "capture_video"]


@dataclass
class DQNArgs:
    exp_name: str = "DQN_implementation"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = t.cuda.is_available()
    log_dir: str = "logs"
    use_wandb: bool = False
    wandb_project_name: str = "CartPoleDQN"
    wandb_entity: Optional[str] = None
    capture_video: bool = True
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500_000
    learning_rate: float = 0.00025
    buffer_size: int = 10_000
    gamma: float = 0.99
    target_network_frequency: int = 500
    batch_size: int = 128
    start_e: float = 1.0
    end_e: float = 0.1
    exploration_fraction: float = 0.2
    train_frequency: int = 10
    log_frequency: int = 50

arg_help_strings = dict(
    exp_name = "the name of this experiment",
    seed = "seed of the experiment",
    torch_deterministic = "if toggled, `torch.backends.cudnn.deterministic=False`",
    cuda = "if toggled, cuda will be enabled by default",
    log_dir = "the name of the logging directory",
    use_wandb = "whether to log to weights and biases",
    wandb_project_name = "the wandb's project name",
    wandb_entity = "the entity (team) of wandb's project",
    capture_video = "whether to capture videos of the agent performances (check out `videos` folder)",
    env_id = "the id of the environment",
    total_timesteps = "total number of steps of the experiments",
    learning_rate = "the learning rate of the optimizer",
    buffer_size = "the replay memory buffer size",
    gamma = "the discount factor gamma",
    target_network_frequency = "the timesteps it takes to update the target network",
    batch_size = "the batch size of samples from the replay memory",
    start_e = "the starting epsilon for exploration",
    end_e = "the ending epsilon for exploration",
    exploration_fraction = "the fraction of `total-timesteps` it takes from start-e to go end-e",
    # learning_starts = "timestep to start learning",
    train_frequency = "number of sampled actions in between each learning step",
    log_frequency = "the frequency of logging"
)

def parse_args(arg_help_strings=arg_help_strings, toggles=toggles):
    parser = argparse.ArgumentParser()
    for (name, field) in DQNArgs.__dataclass_fields__.items():
        flag = "--" + name.replace("_", "-")
        type_function = field.type if field.type != bool else lambda x: bool(strtobool(x))
        toggle_kwargs = {"nargs": "?", "const": True} if name in toggles else {}
        parser.add_argument(
            flag, type=type_function, default=field.default, help=arg_help_strings[name], **toggle_kwargs
        )
    return DQNArgs(**vars(parser.parse_args()))

def plot_buffer_items(df, title):
    fig = px.line(df, facet_row="variable", labels={"value": "", "index": "steps"}, title=title)
    fig.update_layout(template="simple_white")
    fig.layout.annotations = []
    fig.update_yaxes(matches=None)
    fig.show()

def arg_help_DQN(args: Optional[DQNArgs], print_df=False):
    """Prints out a nicely displayed list of arguments, their default values, and what they mean."""
    if args is None:
        args = DQNArgs()
        changed_args = []
    else:
        default_args = DQNArgs()
        changed_args = [key for key in default_args.__dict__ if getattr(default_args, key) != getattr(args, key)]
    df = pd.DataFrame([arg_help_strings]).T
    df.columns = ["description"]
    df["default value"] = [repr(getattr(args, name)) for name in df.index]
    df.index.name = "arg"
    df = df[["default value", "description"]]
    if print_df:
        df.insert(1, "changed?", ["yes" if i in changed_args else "" for i in df.index])
        with pd.option_context(
            'max_colwidth', 0, 
            'display.width', 150, 
            'display.colheader_justify', 'left'
        ):
            print(df)
    else:
        s = (
            df.style
            .set_table_styles([
                {'selector': 'td', 'props': 'text-align: left;'},
                {'selector': 'th', 'props': 'text-align: left;'}
            ])
            .apply(lambda row: ['background-color: red' if row.name in changed_args else None] + [None,] * (len(row) - 1), axis=1)
        )
        with pd.option_context("max_colwidth", 0):
            display(s)