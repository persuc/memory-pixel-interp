from environments.python_breakout import Runner

runner = Runner()

while True:
    runner.step()
    memory = runner.memory()
    if not (memory[0] % 90):
        print(runner.memory())
