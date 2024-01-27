from pathlib import Path
import sys

sys.path.append(str(Path(sys.executable).parent.parent.parent))

import main

from environments.python_breakout import PythonMemoryEnv
from DeepRLA.agents.policy_gradient_agents.PPO import PPO
from utils.agent_config import AgentConfig, policy_gradient_agent_params


class Trainer:
    def __init__(self, agent: PPO, config: AgentConfig) -> None:
        self.agent_config = config
        self.agent = agent

    def train(self):
        """Runs a set of games for a given agent, saving the results in self.results"""
        print("AGENT NAME: {}".format(self.agent.agent_name))
        self.environment_name = self.agent.environment_title
        print(self.agent.hyperparameters)
        print("RANDOM SEED ", self.agent_config.seed)
        game_scores, rolling_scores, time_taken = self.agent.run_n_episodes(
            self.agent_config.num_episodes_to_run
        )
        print(f"{game_scores=} {rolling_scores=} {time_taken=}")
        if self.agent_config.visualise_individual_results:
            # TODO
            pass


# while True:
#     runner.step()
#     memory = runner.memory()
#     if not (memory[0] % 90):
#         print(runner.memory())

env = PythonMemoryEnv(False)
config = AgentConfig(1, env, 100, policy_gradient_agent_params)
agent = PPO(config)
trainer = Trainer(agent, config)
trainer.train()


# from agents.policy_gradient_agents.PPO import PPO
