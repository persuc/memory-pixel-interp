# %%
from dataclasses import dataclass
import gym


@dataclass
class AgentConfig(object):
    """Object to hold the config requirements for an agent/game"""

    seed: int
    environment: gym.Env
    num_episodes_to_run: int
    visualise_overall_results = False
    visualise_individual_results = False
    hyperparameters: dict  # see DeepRLA.agents.Trainer.agent_to_agent_group_dictionary
    requirements_to_solve_game = None
    file_to_save_data_results: str | None = None
    file_to_save_results_graph: str | None = None
    use_GPU = True  # if true, agent uses "cuda:0" else "cpu"
    overwrite_existing_results_file = False
    save_model = False
    standard_deviation_results = 1.0
    randomise_random_seed = True
    show_solution_score = False
    debug_mode = False


# %%

policy_gradient_agent_params = {
    "learning_rate": 0.05,
    "linear_hidden_units": [20, 20],
    "final_layer_activation": "SOFTMAX",
    "learning_iterations_per_round": 5,
    "discount_rate": 0.99,
    "batch_norm": False,
    "clip_epsilon": 0.1,
    "episodes_per_learning_round": 4,
    "normalise_rewards": True,
    "gradient_clipping_norm": 7.0,
    "mu": 0.0,  # only required for continuous action games
    "theta": 0.0,  # only required for continuous action games
    "sigma": 0.0,  # only required for continuous action games
    "epsilon_decay_rate_denominator": 1.0,
    "clip_rewards": False,
}
