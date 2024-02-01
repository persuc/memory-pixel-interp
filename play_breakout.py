
#%%

import gym
import pygame 
from pygame.locals import *
import rich
from rich.console import Console
from rich.table import Column, Table
import time

pygame.init()

env = gym.make("ALE/Breakout-v5", render_mode="human", obs_type="ram")
env = env.unwrapped
env.reset()
action = 0
running = True
print("Breakout-v5")
print("Left/Right to control Paddle")
print("F to fire ball")
print("Q to quit, R to reset")
env.metadata['render_fps'] = 60
actions = [1, 0, 0, 0]
action_idx = 0

important_idxs = set()

initial_ram = []
current_ram = [0] * 256
previous_ram = [0] * 256
step = 0
console = Console()
special_numbers = set()
special_metric = [0] * 128
max_specialness = 10

# %%
def compare(a, b):
    a = bin(a)[2:].zfill(8)
    b = bin(b)[2:].zfill(8)
    is_one_diff = sum(a[i] != b[i] for i in range(8)) == 1
    # print(a + '\n' + b + (" OFF BY ONE" if is_one_diff else ""))
    return is_one_diff

while action_idx < len(actions):
    
    keys = pygame.key.get_pressed()
    if keys[K_LEFT]:
        action = 3
    elif keys[K_RIGHT]:
        action = 2
    elif keys[K_f]:
        action = 1
    elif keys[K_r]:
        env.reset()
    elif keys[K_q]:
        break
    else:
        action = 0
    # action = actions[action_idx]
    # action_idx += 1
    obs, _, done, truncated, _ = env.step(action)
    current_ram = obs

    time.sleep(0.25)

    # if action_idx == 0:
    #     action_idx += 1
    #     initial_ram = obs
    # else:
    #     for i, v in enumerate(obs):
    #         if v != initial_ram[i]:
    #             important_idxs.add(v)


    def format_memory(idx: int):
        if idx >= 128:
            return ""
        # if idx in special_numbers:
        #     return f"[bold #00ffff]{obs[idx]}[/bold #00ffff]"
        # if special_metric[idx]:
        if special_metric[idx] > 0:
            specialness = round(special_metric[idx] / max_specialness * 255)
            hex_clr = hex(specialness)[2:].zfill(2)
            color = f"#00{hex_clr}{hex_clr}"
            # print(color)
            return f"[bold {color}]{obs[idx]}[/bold {color}]"
        # if obs[idx] < previous_ram[idx]:
        #     return f"[bold red]{obs[idx]}[/bold red]"
        # elif obs[idx] > previous_ram[idx]:
        #   return f"[bold green]{obs[idx]}[/bold green]"

        return str(obs[idx])
        

    # input("Press Enter to continue...")
    
    # table = Table(*[str(i) for i in range(20)])
    # for i in range(13):
    #   table.add_row(
    #     *[format_memory(i) for i in range(i * 20, (i + 1) * 20)]
    #   )
    
    # console.clear()
    # console.print(table)

    # for i in range(128):
    #     if compare(obs[i], previous_ram[i]):
    #         special_numbers.add(i)
    #         special_metric[i] = min(special_metric[i], max_specialness)

    previous_ram = obs
    # rich.print(table, flush=True)
    # console.print(table
    
        
    # print(obs)
    if done:
       env.reset()

    step += 1


for i in range(128):
    if obs[i] != previous_ram[i]:
        print(f"{i}: {previous_ram[i]} -> {obs[i]}")


# print(special_numbers)
# print(special_metric)

# print(list(important_idxs))

#%%
# recorded_important_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 15, 16, 18, 20, 22, 24, 28, 32, 34, 36, 40, 44, 46, 48, 52, 55, 56, 57, 58, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 74, 76, 77, 78, 80, 82, 83, 84, 85, 86, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 118, 119, 120, 121, 122, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 136, 137, 138, 139, 140, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 154, 155, 156, 157, 158, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 180, 181, 182, 184, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 198, 199, 200, 202, 204, 205, 206, 208, 212, 216, 220, 224, 228, 232, 236, 240, 243, 244, 246, 248, 252, 254, 255]

# print([i for i in range(256) if i not in recorded_important_idxs])
# %%

# 69: distance from right wall (0-255)
# 71: distance from left wall (0-255)
# 72: counter
# 99: ball x pos (0-255)
# 101: ball y pos (0-255) bottom of screen ~= 210
# 76: score (0-max score)
    



# │ 63  │ 63  │ 63  │ 63  │ 63  │ 63  │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │
# │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 192 │ 192 │ 192 │ 192 │ 192 │ 192 │ 255 │ 255 │ 255 │ 255 │
# │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 240 │ 0   │ 0   │ 255 │ 0   │ 0   │ 240 │ 0   │ 5   │ 0   │ 0   │
# │ 6   │ 0   │ 70  │ 182 │ 134 │ 198 │ 22  │ 38  │ 54  │ 70  │ 88  │ 6   │ 146 │ 0   │ 4   │ 13  │ 0   │ 0   │ 0   │ 0   │
# │ 0   │ 241 │ 0   │ 242 │ 0   │ 242 │ 25  │ 241 │ 5   │ 242 │ 24  │ 5   │ 255 │ 0   │ 0   │ 0   │ 0   │ 0   │ 0   │ 0   │
# │ 0   │ 0   │ 0   │ 0   │ 0   │ 0   │ 0   │ 0   │ 8   │ 0   │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 0   │ 0   │ 255 │
# │ 0   │ 0   │ 186 │ 214 │ 117 │ 246 │ 219 │ 242 |

# 18, 77, 96, 99, 91, 90, 70, 107, 105, 84, 104, 103, 102, 122, 101, 121

# │ 63  │ 63  │ 63  │ 63  │ 63  │ 63  │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 243 │ 255 │
# │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 192 │ 192 │ 192 │ 192 │ 192 │ 192 │ 255 │ 255 │ 255 │ 255 │
# │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 240 │ 0   │ 0   │ 255 │ 0   │ 0   │ 240 │ 0   │ 5   │ 0   │ 0   │
# │ 6   │ 0   │ 70  │ 182 │ 134 │ 198 │ 22  │ 38  │ 54  │ 70  │=184=│ 6   │=55 =│ 0   │ 4   │ 13  │ 0   │=1  =│ 0   │ 0   │
# │ 0   │ 241 │ 0   │ 242 │=5  =│ 242 │ 25  │ 241 │ 5   │ 242 │=184=│=0  =│ 255 │ 0   │ 0   │ 0   │=28 =│ 0   │ 0   │=124=│
# │ 0   │=109=│=128=│=1  =│=128=│=255=│ 0   │=129=│ 8   │=2  =│ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 255 │ 0   │ 0   │ 255 │
# │ 0   │=150=│=150=│ 214 │ 117 │ 246 │ 219 │ 242 │


# %%
