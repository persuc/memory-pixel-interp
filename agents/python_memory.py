from pathlib import Path
import pickle
import sys
import time
from typing import Literal
from gym.wrappers.record_video import RecordVideo
import torch
import random
from random import randint
import wandb
from contextlib import closing
from torch.multiprocessing import Pool
from nn_builder.pytorch.NN import NN

sys.path.append(str(Path.cwd()))

import main

from environments.python_breakout import PythonMemoryEnv
from DeepRLA.utilities.Utility_Functions import normalise_rewards
from DeepRLA.agents.policy_gradient_agents.PPO import PPO as DeepRLAPPO
from DeepRLA.utilities.Parallel_Experience_Generator import Parallel_Experience_Generator as DeepRLA_Parallel_Experience_Generator
from DeepRLA.utilities.Utility_Functions import create_actor_distribution
from utils.agent_config import AgentConfig, policy_gradient_agent_params


class Parallel_Experience_Generator(DeepRLA_Parallel_Experience_Generator):
    environment: PythonMemoryEnv
    policy: NN

    def play_n_episodes(self, n, exploration_epsilon=None):
        """Plays n episodes in parallel using the fixed policy and returns the data"""
        self.exploration_epsilon = exploration_epsilon
        results = [self(n) for _ in range(n)]
        states_for_all_episodes = [episode[0] for episode in results]
        actions_for_all_episodes = [episode[1] for episode in results]
        rewards_for_all_episodes = [episode[2] for episode in results]
        return states_for_all_episodes, actions_for_all_episodes, rewards_for_all_episodes

    def play_1_episode(self, epsilon_exploration: float):
        """Plays 1 episode using the fixed policy and returns the data"""
        state: list[int | float]
        state, _info = self.reset_game()
        # 
        done = False
        episode_states = []
        episode_actions = []
        episode_rewards = []
        while not done:
            action: Literal[0, 1, 2] = self.pick_action(state, epsilon_exploration)
            direction: Literal["left", "right", "none"] = self.environment._action_to_direction[action]
            next_state, reward, done, _info = self.environment.step(direction)
            if self.hyperparameters["clip_rewards"]: reward = max(min(reward, 1.0), -1.0)
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            state = next_state
            # print(f"{state=}")
        return episode_states, episode_actions, episode_rewards
    
    def pick_action(self, state: list[int | float], epsilon_exploration: float) -> Literal[0, 1, 2]:
        """Picks an action using the policy"""
        if self.action_types == "DISCRETE":
            if random.random() <= epsilon_exploration:
                action = random.randint(0, self.action_size - 1)
                return action
        device = 'cuda:0' if self.use_GPU else 'cpu'
        state = torch.tensor(state).float().unsqueeze(0).to(device)
        actor_output = self.policy.forward(state)
        if self.action_choice_output_columns is not None:
            actor_output = actor_output[:, self.action_choice_output_columns]
        action_distribution = create_actor_distribution(self.action_types, actor_output, self.action_size)
        action = action_distribution.sample().cpu()

        if self.action_types == "CONTINUOUS": action += torch.Tensor(self.noise.sample())
        else: action = action.item()
        return action

class PPO(DeepRLAPPO):
    """
        This class overrides some methods in the Deep RLA agent
        in order to fix some bugs related to compatibility
        with newer versions of gym
    """

    def __init__(self, config):
        super().__init__(config)

        self.experience_generator = Parallel_Experience_Generator(self.environment, self.policy_new, self.config.seed,
                                                                  self.hyperparameters, self.action_size, use_GPU=config.use_GPU)

        # TODO: DeepRLA PPO does not support mps
        # if not config.use_GPU:
        #     return
        # if torch.cuda.is_available():
        #     self.device = 'cuda:0'
        # if torch.backends.mps.is_available():
        #     self.device = 'mps'

    def get_state_size(self):
        """Gets the state_size for the gym env into the correct shape for a neural network"""
        # TODO: make custom superclass for this project's environments with
        #       get_state_size method
        return self.environment.state_size
    
    def get_trials(self):
        """Gets the number of trials to average a score over"""
        return 100

class RecordVideoSerializable(RecordVideo):
    def __init__(
        self,
        env,
        video_folder: str,
        episode_trigger: int | None = None,
        step_trigger: int | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
    ):
        super().__init__(env=env, video_folder=video_folder, video_length=video_length, name_prefix=name_prefix)
        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self._action_to_direction = env._action_to_direction

    def _video_enabled(self):
        if self.step_trigger:
            return self.step_id % self.step_trigger == 0
        elif self.episode_trigger:
            return self.episode_id % self.episode_trigger == 0
        else:
            raise Exception(f"must provide either step trigger or episode trigger")


class Trainer:
    def __init__(self, agent: PPO, config: AgentConfig) -> None:
        self.agent_config = config
        self.agent = agent
        self.loss_history: list[torch.Tensor] = []

    def step_agent(self):
        """Runs a step for the PPO agent"""
        exploration_epsilon =  self.agent.exploration_strategy.get_updated_epsilon_exploration({"episode_number": self.agent.episode_number})
        self.agent.many_episode_states, self.agent.many_episode_actions, self.agent.many_episode_rewards = self.agent.experience_generator.play_n_episodes(
            self.agent.hyperparameters["episodes_per_learning_round"], exploration_epsilon)
        self.agent.episode_number += self.agent.hyperparameters["episodes_per_learning_round"]
        self.policy_learn()
        self.agent.update_learning_rate(self.agent.hyperparameters["learning_rate"], self.agent.policy_new_optimizer)
        self.agent.equalise_policies()

    def policy_learn(self):
        """A learning iteration for the policy"""
        all_discounted_returns = self.agent.calculate_all_discounted_returns()
        if self.agent.hyperparameters["normalise_rewards"]:
            all_discounted_returns = normalise_rewards(all_discounted_returns)
        for _ in range(self.agent.hyperparameters["learning_iterations_per_round"]):
            all_ratio_of_policy_probabilities = self.agent.calculate_all_ratio_of_policy_probabilities()
            loss = self.agent.calculate_loss([all_ratio_of_policy_probabilities], all_discounted_returns)
            self.loss_history.append(loss)
            self.agent.take_policy_new_optimisation_step(loss)

    def train(
        self, save_result=True, print_result=True, show_whether_achieved_goal=False, use_wandb=False
    ):
        """Runs a set of games for a given agent, saving the results in self.results"""
        print("AGENT NAME: {}".format(self.agent.agent_name))
        self.environment_name = self.agent.environment_title
        if use_wandb:
            wandb.init(monitor_gym=self.agent_config.capture_video)
        itr_per_round = self.agent.hyperparameters["learning_iterations_per_round"]
        self.loss_history = []
        print(self.agent.hyperparameters)
        print("RANDOM SEED ", self.agent_config.seed)

        round = 0
        start = time.time()
        while round < self.agent_config.num_episodes_to_run:
            self.agent.reset_game()
            self.step_agent()
            avg_loss = sum(self.loss_history[-itr_per_round:]) / itr_per_round
            if save_result:
                self.agent.save_result()
            if print_result:
                self.agent.print_rolling_result()
                print(f"loss: {avg_loss}")
            if use_wandb:
                wandb.log({"loss": avg_loss, })

            if isinstance(self.agent_config.save_model_path, str):
                torch.save({
                    "round": round,
                    "model_state_dict": self.agent.policy_new.state_dict(),
                    "optimizer_state_dict": self.agent.policy_new_optimizer.state_dict(),
                    "train_loss_history": self.loss_history,
                }, self.agent_config.save_model_path)
            round += 1
        time_taken = time.time() - start
        if show_whether_achieved_goal:
            self.agent.show_whether_achieved_goal()

        print(
            f"game_scores={self.agent.game_full_episode_scores} rolling_scores={self.agent.rolling_results} {time_taken=}"
        )
        if self.agent_config.visualise_individual_results:
            # TODO
            pass
        if use_wandb:
            wandb.finish()

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    # torch.multiprocessing.set_start_method('fork')
    env = PythonMemoryEnv(render_mode="rgb_array")
    
    env = RecordVideoSerializable(
        env, 
        f"videos/test",
        episode_trigger = 10
    )
    
    config = AgentConfig(
        seed=1,
        environment=env,
        num_episodes_to_run=100,
        hyperparameters=policy_gradient_agent_params,
        save_model_path="./PythonMemory.pt",
    )
    agent = PPO(config)
    trainer = Trainer(agent, config)
    trainer.train(print_result=True, save_result=True, use_wandb=True)
