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

import math, pygame, sys, shutil, getpass
from pygame.locals import *

pygame.init()
fpsClock = pygame.time.Clock()
screen = pygame.display.set_mode((640, 480))  # create screen - 640 pix by 480 pix
pygame.display.set_caption("Breakout")  # set title bar

# generic colors-------------------------------
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)
white = pygame.Color(255, 255, 255)
grey = pygame.Color(142, 142, 142)
black = pygame.Color(0, 0, 0)

# row colors-----------------------------------
r1 = pygame.Color(200, 72, 72)
r2 = pygame.Color(198, 108, 58)
r3 = pygame.Color(180, 122, 48)
r4 = pygame.Color(162, 162, 42)
r5 = pygame.Color(72, 160, 72)
r6 = pygame.Color(67, 73, 202)
colors = [r1, r2, r3, r4, r5, r6]

# variables------------------------------------
controls = "keys"  # control method
dx, dy = 18, 6  # dimensions of board
bx, by = 50, 150  # board position
wall1 = pygame.Rect(20, 100, 30, 380)  # walls of the game
wall2 = pygame.Rect(590, 100, 30, 380)
wall3 = pygame.Rect(20, 80, 600, 30)


# Creates a board of rectangles----------------
def new_board():
    board = []
    for x in range(dx):
        board.append([])
        for y in range(dy):
            board[x].append(1)
    return board


# Classes defined------------------------------
class Paddle:  # class for paddle vars
    x = 320
    y = 450
    size = 2  # 2 is normal size, 1 is half-size
    direction = "none"


class Ball:  # class for ball vars
    x = 0
    y = 0
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


# Functions defined----------------------------
def print_board(board, colors):  # prints the board
    for x in range(dx):
        for y in range(dy):
            if board[x][y] == 1:
                pygame.draw.rect(
                    screen, colors[y], (((x * 30) + bx), ((y * 12) + by), 30, 12)
                )


def print_paddle(paddle):  # prints the paddle
    if paddle.size == 2:
        pygame.draw.rect(screen, red, ((paddle.x - 20), (paddle.y), 40, 5))


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


def write(x, y, color, msg, font):  # prints onto the screen in selected font
    msgSurfaceObj = font.render(msg, False, color)
    msgRectobj = msgSurfaceObj.get_rect()
    msgRectobj.topleft = (x, y)
    screen.blit(msgSurfaceObj, msgRectobj)


def game(score, paddle, ball, board, wall1, font, colO, colR):  # The game itself
    # Draw all the things------------------------------
    screen.fill(black)
    pygame.draw.rect(screen, grey, wall1)
    pygame.draw.rect(screen, grey, wall2)
    pygame.draw.rect(screen, grey, wall3)
    pygame.draw.rect(screen, red, (ball.x - 3, ball.y - 3, 6, 6))
    print_board(board, colors)
    print_paddle(paddle)
    write(20, 20, grey, str(score), font)
    temp = 0
    for life in range(ball.remaining):
        if life != 0:
            pygame.draw.rect(screen, red, (600, 400 - temp, 10, 10))
            temp += 15

    # check all the collisions-------------------------
    if ball.adjusted == False:
        ball.adjust()
    ball.x += ball.xPos
    ball.y += ball.yPos
    if ball.y < 455 and ball.y > 445:
        if ball.x > paddle.x - 20 and ball.x < paddle.x + 20:
            ball.adjusted, ball.xPos, ball.yPos = collide_paddle(
                paddle, ball
            )  # paddle collide
            ball.collisions += 1
            # increase ball speeds at 4 hits on paddle, 12 hits, orange row, red row
            if ball.collisions == 4:
                ball.speed += 1
            if ball.collisions == 12:
                ball.speed += 1
            # if ball hits the back wall, paddle cuts in half
    # check wall collide----------------------------
    if wall1.collidepoint(ball.x, ball.y) == True or wall2.collidepoint(ball.x, ball.y):
        ball.xPos = -(ball.xPos)
    if wall3.collidepoint(ball.x, ball.y) == True:
        ball.yPos = -(ball.yPos)

    # check collision with bricks-------------------
    Break = False
    for x in range(dx):
        for y in range(dy):
            if board[x][y] == 1:
                block = pygame.Rect(30 * x + bx - 1, 12 * y + by - 1, 32, 14)
                if block.collidepoint(ball.x, ball.y) == True:
                    board[x][y] = 0
                    ##                            if y*12+by+12 < ball.y: FIX THIS ITS THE BLOCK BUG
                    ##                                ball.y = -(ball.y)
                    ##                            elif x*30+bx+30 <
                    ball.yPos = -ball.yPos  # Cheat
                    if y == 4 or y == 5:
                        score += 1
                    elif y == 2 or y == 3:
                        score += 4
                        if colO == False:
                            colO = True
                            ball.speed += 1
                    else:
                        score += 7
                        if colR == False:
                            colR = True
                            ball.speed += 2
                    Break = True
            if Break == True:
                break
        if Break == True:
            break
    if ball.y > 460:
        ball.alive = False

    # check if ball was lost
    if ball.alive == False:
        ball.remaining -= 1

    # move paddle
    if paddle.direction == "right":
        if paddle.x <= 561:
            paddle.x += 8
    elif paddle.direction == "left":
        if paddle.x >= 79:
            paddle.x -= 8

    # get user input
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == KEYDOWN:
            if event.key == K_LEFT:
                paddle.direction = "left"
            if event.key == K_RIGHT:
                paddle.direction = "right"
        elif event.type == KEYUP:
            if event.key == K_LEFT:
                if paddle.direction == "left":
                    paddle.direction = "none"
            if event.key == K_RIGHT:
                if paddle.direction == "right":
                    paddle.direction = "none"

    # update display
    pygame.display.update()
    fpsClock.tick(30)
    return score


class Runner:
    def __init__(self):
        self.reset()
        self.fontObj = pygame.font.Font("freesansbold.ttf", 24)

    def reset(self):
        self.board = new_board()
        self.score = 0
        self.paddle = Paddle()
        self.ball = Ball()
        screen.fill(black)
        self.steps = 0

    def step(self):
        colO = False  # check collision with the orange row, for speed purposes
        colR = False  # same but for red row
        self.score = game(
            self.score,
            self.paddle,
            self.ball,
            self.board,
            wall1,
            self.fontObj,
            colO,
            colR,
        )

        self.steps += 1

        if self.ball.remaining == 0:
            pygame.quit()
            sys.exit()

        if not self.ball.alive:
            # starting variables
            self.ball.alive = True
            self.ball.x = 53
            self.ball.y = 300
            self.ball.collisions, self.ball.speed = 0, 5
            colO = False
            colR = False
            self.ball.speed = 5
            self.ball.xPos = 1
            self.ball.yPos = 1
            self.ball.adjusted = False

    def memory(self):
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


# -----------------------------------------------------
if __name__ == "__main__":
    runner = Runner()
    while True:
        runner.step()
