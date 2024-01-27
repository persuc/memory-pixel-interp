from pathlib import Path
import sys

# add every bloody folder to the path so that you can import anything from anywhere
interpreter_path = Path(sys.executable)
project_path = interpreter_path.parent.parent.parent
sys.path.append("./DeepRLA/")

# include this at the top of each file

# from pathlib import Path
# import sys

# sys.path.append(str(Path(sys.executable).parent.parent.parent))

# import main
