from pathlib import Path
import sys

sys.path.append(str(Path(sys.executable).parent.parent.parent))

from environments.python_breakout import Runner

runner = Runner()

while True:
    runner.step()
    memory = runner.memory()
    if not (memory[0] % 90):
        print(runner.memory())


# from agents.policy_gradient_agents.PPO import PPO
