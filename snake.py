import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font("asset/PressStart2P-Regular.ttf", 17)

# rgb colors
WHITE1 = (255, 255, 255)
WHITE2 = (200, 200, 200)
RED = (200, 0, 0)
BLACK = (0, 0, 0)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 200, 0)

BLOCK_SIZE = 20
SPEED = 2000


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"


def replay(play_again_rect):
    # wait for the user to click play again
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            # Reset if Mouse Click is close to play again text
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if play_again_rect.collidepoint(mouse_pos):
                    return


class SnakeAI:
    def __init__(self, w=1040, h=680):
        self.w = w
        self.h = h

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        
        #Initialize game state
        self.reset()    

    def _place_food(self):  #? note: _ mean private class/method
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)

        if self.food in self.snake:
            self._place_food()

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # hit boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        is_head = True
        self.display.fill(BLACK)

        for pt in self.snake:
            if is_head:
                pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
                is_head = False
            else:
                pygame.draw.rect(self.display, WHITE2, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, WHITE1, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render(str(self.score), True, WHITE1)
        text_rect = text.get_rect(center=(self.w / 2, 25))
        self.display.blit(text, text_rect)
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = clock_wise.index(self.direction)  # track snake direction in clockwise

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[index]  # no change
        elif np.array_equal(action, [0, 1, 0]):  # right [0, 1, 0]
            next_index = (index + 1) % 4
            new_dir = clock_wise[next_index]  # right turn r -> d -> l -> u
        else:  # left [0, 0, 1]
            next_index = (index - 1) % 4
            new_dir = clock_wise[next_index]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT  # Ini Direction
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _render_text(self, text, font, color, center):
        rendered_text = font.render(text, True, color)
        text_rect = rendered_text.get_rect(center=center)
        self.display.blit(rendered_text, text_rect)
        return text_rect

    def show_game_over_screen(self, score):
        self.display.fill(BLACK)

        self._render_text("GAME OVER", font, RED, (self.w / 2, self.h / 2 - 50))
        self._render_text("Score: " + str(score), font, WHITE1, (self.w / 2, self.h / 2))

        play_again_rect = self._render_text("Play Again", font, BLACK, (self.w / 2, self.h / 2 + 50))
        play_again_text = font.render("Play Again", True, BLACK)
        button_rect = play_again_rect.inflate(20, 10)  # padding around the text
        pygame.draw.rect(self.display, WHITE1, button_rect)
        self.display.blit(play_again_text, play_again_rect)

        pygame.display.flip()

        replay(play_again_rect)

    def play(self, action):
        self.frame_iteration += 1

        # 1. Check if we want to keep the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Update Snake Moment and Size by BLOCK
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        # Check the length of the snake through frame_iteration incase it doesn't grow
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

            # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        return reward, game_over, self.score
