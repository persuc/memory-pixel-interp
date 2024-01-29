# Credit https://github.com/aknuck/Atari-Breakout/

#####################################
#          ATARI BREAKOUT           #
#                                   #
#          Python code by           #
#           Adam Knuckey            #
#               2013                #
#                                   #
#    Original Game by Atari, inc    #
#                                   #
#  Controls:                        #
#  - arrow keys: move paddle        #
#                                   #
#  Scoring:                         #
#  - Green / blue blocks ........ 1 #
#  - Yellow / gold blocks ....... 4 #
#  - orange / red blocks ........ 7 #
#                                   #
#####################################

import math, pygame, sys
import numpy
import gym
from gym import spaces
from gym.envs.registration import EnvSpec
from typing import Any, Literal, Mapping

# variables------------------------------------
bx, by = 50, 150  # board position


# Classes defined------------------------------
class Paddle:  # class for paddle vars
    x = 320
    y = 450
    size = 2  # 2 is normal size, 1 is half-size
    direction: Literal["left", "right", "none"] = "none"


class Ball:  # class for ball vars
    x: float = 0
    y: float = 0
    remaining = 3
    xPos = 1  # amount increasing by for x. adjusted for speed
    yPos = 1
    adjusted = False  # says wether the xPos and yPos have been adjusted for speed
    speed = 5
    collisions = 0
    alive = False

    def adjust(
        self,
    ):  # adjusts the x and y being added to the ball to make the hypotenuse the ball speed
        tSlope = math.sqrt(self.xPos**2 + self.yPos**2)
        self.xPos = (self.speed / tSlope) * self.xPos
        self.yPos = (self.speed / tSlope) * self.yPos
        self.adjusted = True

    def reset(self, rand: float):
        self.x = 53 + 480 * rand
        self.y = 300
        self.xPos = 1
        self.yPos = 1
        self.adjusted = False
        self.speed = 5
        self.collisions, self.speed = 0, 5
        self.alive = True


# Functions defined----------------------------
def print_board(board, colors, screen):  # prints the board
    for x in range(len(board)):
        for y in range(len(board[0])):
            if board[x][y] == 1:
                pygame.draw.rect(
                    screen, colors[y], (((x * 30) + bx), ((y * 12) + by), 30, 12)
                )


def print_paddle(paddle, screen, color):  # prints the paddle
    if paddle.size == 2:
        pygame.draw.rect(screen, color, ((paddle.x - 20), (paddle.y), 40, 5))


def collide_paddle(
    paddle, ball
):  # recalculates the trajectory for the ball after collision with the paddle
    ball.adjusted = False
    if ball.x - paddle.x != 0:
        ball.xPos = (ball.x - paddle.x) / 8
        ball.yPos = -1
    else:
        ball.xPos = 0
        ball.yPos = 1
    return ball.adjusted, float(ball.xPos), float(ball.yPos)


def write(x, y, color, msg, font, screen):  # prints onto the screen in selected font
    msgSurfaceObj = font.render(msg, False, color)
    msgRectobj = msgSurfaceObj.get_rect()
    msgRectobj.topleft = (x, y)
    screen.blit(msgSurfaceObj, msgRectobj)

def do_render(screen: pygame.surface.Surface, colors, row_colors, board, ball, paddle, wall1, wall2, wall3, score=None, fontObj=None):
    screen.fill(colors["black"])
    pygame.draw.rect(screen, colors["grey"], wall1)
    pygame.draw.rect(screen, colors["grey"], wall2)
    pygame.draw.rect(screen, colors["grey"], wall3)
    pygame.draw.rect(
        screen,
        colors["red"],
        (ball.x - 3, ball.y - 3, 6, 6),
    )
    print_board(board, row_colors, screen)
    print_paddle(paddle, screen, colors["red"])
    if fontObj is not None:
        write(
            20, 20, colors["grey"], str(score), fontObj, screen
        )
    temp = 0
    for life in range(ball.remaining):
        if life != 0:
            pygame.draw.rect(
                screen, colors["red"], (600, 400 - temp, 10, 10)
            )
            temp += 15

class Runner:
    render_mode: Literal["human", "rgb_array", None]
    board_width = 18
    board_height = 6
    screen_width = 640
    screen_height = 480
    ball_start_x: numpy.ndarray[Any, numpy.dtypes.Float64DType]
    lives = 3

    def __init__(self, render_mode: Literal["human", "rgb_array", None]):
        self.render_mode = render_mode
        self.ball_start_x = numpy.random.rand(self.lives)

        pygame.init()
        if self.render_mode is not None:
            if self.render_mode == "human":
                self.fontObj = pygame.font.Font("freesansbold.ttf", 24)
                self.fpsClock = pygame.time.Clock()
                pygame.display.set_caption("Breakout")  # set title bar
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

            self.colors = {
                # generic colors-------------------------------
                "red": pygame.Color(255, 0, 0),
                "green": pygame.Color(0, 255, 0),
                "blue": pygame.Color(0, 0, 255),
                "white": pygame.Color(255, 255, 255),
                "grey": pygame.Color(142, 142, 142),
                "black": pygame.Color(0, 0, 0),
                # row colors-----------------------------------
                "r1": pygame.Color(200, 72, 72),
                "r2": pygame.Color(198, 108, 58),
                "r3": pygame.Color(180, 122, 48),
                "r4": pygame.Color(162, 162, 42),
                "r5": pygame.Color(72, 160, 72),
                "r6": pygame.Color(67, 73, 202),
            }

            self.row_colors = [
                self.colors["r1"],
                self.colors["r2"],
                self.colors["r3"],
                self.colors["r4"],
                self.colors["r5"],
                self.colors["r6"],
            ]

        # world objects
        self.wall1 = pygame.Rect(20, 100, 30, 380)
        self.wall2 = pygame.Rect(590, 100, 30, 380)
        self.wall3 = pygame.Rect(20, 80, 600, 30)

        self.reset()

    def seed(self, seed: int) -> None:
        numpy.random.seed(seed)
        self.ball_start_x = numpy.random.rand(self.lives)
        self.reset()

    def reset(self):
        # TODO: the board is laid out height x width in memory
        # but displayed as width x height in game
        self.board = []
        for x in range(self.board_width):
            self.board.append([])
            for _ in range(self.board_height):
                self.board[x].append(1)
        self.score = 0
        self.paddle = Paddle()
        self.ball = Ball()
        self.ball.reset(self.ball_start_x[self.ball.remaining - 1])
        self.steps = 0

    def step(self, direction: Literal["left", "right", "none"]):
        colO = False  # check collision with the orange row, for speed purposes
        colR = False  # same but for red row
        self.paddle.direction = direction
        self.game(colO, colR)

        self.steps += 1

        if self.ball.remaining == 0:
            pygame.quit()

        if not self.ball.alive:
            colO = False
            colR = False
            self.ball.reset(self.ball_start_x[self.ball.remaining - 1])


    def game(self, colO, colR):
        if self.render_mode == "human":
            do_render(self.screen, self.colors, self.row_colors, self.board, self.ball, self.paddle, self.wall1, self.wall2, self.wall3, self.score, self.fontObj)

        # check all the collisions-------------------------
        if self.ball.adjusted == False:
            self.ball.adjust()
        self.ball.x += self.ball.xPos
        self.ball.y += self.ball.yPos
        if self.ball.y < 455 and self.ball.y > 445:
            if self.ball.x > self.paddle.x - 20 and self.ball.x < self.paddle.x + 20:
                self.ball.adjusted, self.ball.xPos, self.ball.yPos = collide_paddle(
                    self.paddle, self.ball
                )  # paddle collide
                self.ball.collisions += 1
                # increase ball speeds at 4 hits on paddle, 12 hits, orange row, red row
                if self.ball.collisions == 4:
                    self.ball.speed += 1
                if self.ball.collisions == 12:
                    self.ball.speed += 1
                # if ball hits the back wall, paddle cuts in half
        # check wall collide----------------------------
        if self.wall1.collidepoint(
            self.ball.x, self.ball.y
        ) == True or self.wall2.collidepoint(self.ball.x, self.ball.y):
            self.ball.xPos = -(self.ball.xPos)
        if self.wall3.collidepoint(self.ball.x, self.ball.y) == True:
            self.ball.yPos = -(self.ball.yPos)

        # check collision with bricks-------------------
        Break = False
        for x in range(self.board_width):
            for y in range(self.board_height):
                if self.board[x][y] == 1:
                    block = pygame.Rect(30 * x + bx - 1, 12 * y + by - 1, 32, 14)
                    if block.collidepoint(self.ball.x, self.ball.y) == True:
                        self.board[x][y] = 0
                        ##                            if y*12+by+12 < self.ball.y: FIX THIS ITS THE BLOCK BUG
                        ##                                self.ball.y = -(self.ball.y)
                        ##                            elif x*30+bx+30 <
                        self.ball.yPos = -self.ball.yPos  # Cheat
                        if y == 4 or y == 5:
                            self.score += 1
                        elif y == 2 or y == 3:
                            self.score += 4
                            if colO == False:
                                colO = True
                                self.ball.speed += 1
                        else:
                            self.score += 7
                            if colR == False:
                                colR = True
                                self.ball.speed += 2
                        Break = True
                if Break == True:
                    break
            if Break == True:
                break
        if self.ball.y > 460:
            self.ball.alive = False

        # check if ball was lost
        if self.ball.alive == False:
            self.ball.remaining -= 1

        # move paddle
        if self.paddle.direction == "right":
            if self.paddle.x <= 561:
                self.paddle.x += 8
        elif self.paddle.direction == "left":
            if self.paddle.x >= 79:
                self.paddle.x -= 8

        if self.render_mode is not None:
            if self.render_mode == "human":
                pygame.display.update()
                self.fpsClock.tick(30)

    def get_memory(self) -> list[int | float]:
        flat_board = []
        for row in self.board:
            flat_board.extend(row)
        return [
            self.steps,
            *flat_board,
            self.score,
            self.paddle.x,
            self.paddle.y,
            self.ball.x,
            self.ball.y,
            self.ball.remaining,
            self.ball.xPos,
            self.ball.yPos,
            self.ball.adjusted,
            self.ball.speed,
        ]

class PythonMemoryEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env = "arena-3/PythonBreakoutMemory-v0"
    spec: EnvSpec
    state_size: int
    render_mode: Literal["human", "rgb_array", None]

    def __init__(self, render_mode=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"], f"render_mode must be None, \"human\", or \"rgb_array\""
        self.render_mode = render_mode
        self.runner = Runner(render_mode=self.render_mode)
        self.current_score = 0

        board_size = self.runner.board_width * self.runner.board_height
        self.state_size = board_size + 11

        max_reward = self.runner.board_width * 2 * 1 + self.runner.board_width * 2 * 4 + self.runner.board_width * 2 * 7

        self.spec = EnvSpec(id_requested=self.env, entry_point='environments.python_breakout:PythonMemoryEnv', reward_threshold=max_reward, max_episode_steps=300)
        
        self.observation_space = spaces.Discrete(self.state_size)
        
        # We have 2 actions, corresponding to right, left, none
        self.action_space = spaces.Discrete(3)

        self._action_to_direction: Mapping[Literal[0, 1, 2], Literal["left", "right", "none"]] = {
            0: "right",
            1: "left",
            2: "none"
        }

    def _get_obs(self):
        return self.runner.get_memory()

    def _get_info(self):
        return {}
    
    def seed(self, seed: int) -> None:
        self.runner.seed(seed)

    def reset(self, seed=None, options=None):
        self.runner.reset()
        self.current_score = self.runner.score
        return self._get_obs(), self._get_info()

    def step(self, action: Literal["left", "right", "none"]):
        self.runner.render_mode = self.render_mode
        if self.runner.ball.remaining:
            self.runner.step(action)
            reward = self.runner.score - self.current_score
            self.current_score = self.runner.score
        else:
            reward = 0
        # obs, reward, done, info
        return (
            self._get_obs(),
            reward,
            self.runner.ball.remaining > 0,
            self._get_info(),
        )

    def render(self, mode: Literal["human", "rgb_array", None]):
        if mode != "rgb_array":
            raise Exception(f"Requested render but environment was initialized with render_mode={self.render_mode}")
        if self.render_mode != "rgb_array":
            raise Exception(f"Requested rgb_array but environment was initialized with render_mode={self.render_mode}")
        screen = pygame.Surface((self.runner.screen_width, self.runner.screen_height))
        do_render(screen, self.runner.colors, self.runner.row_colors, self.runner.board, self.runner.ball, self.runner.paddle, self.runner.wall1, self.runner.wall2, self.runner.wall3)
        return numpy.transpose(
            numpy.array(pygame.surfarray.pixels3d(screen)).copy(), axes=(1, 0, 2)
        )


# -----------------------------------------------------
if __name__ == "__main__":
    runner = Runner(render_mode="human")
    runner.seed(0)
    while runner.ball.remaining > 0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    runner.paddle.direction = "left"
                if event.key == pygame.K_RIGHT:
                    runner.paddle.direction = "right"
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    if runner.paddle.direction == "left":
                        runner.paddle.direction = "none"
                if event.key == pygame.K_RIGHT:
                    if runner.paddle.direction == "right":
                        runner.paddle.direction = "none"
        runner.step(runner.paddle.direction)
