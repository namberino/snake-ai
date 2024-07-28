import pygame
import random
from enum import Enum

pygame.init()
font = pygame.font.Font("asset/PressStart2P-Regular.ttf", 17)

# directions
RIGHT = 1
LEFT = 2
UP = 3
DOWN = 4

# rgb colors
WHITE1 = (255, 255, 255)
WHITE2 = (200, 200, 200)
RED = (200, 0, 0)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

BLOCK_SIZE = 20
SPEED = 10

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

class SnakeGame:
    def __init__(self, w=1040, h=680):
        self.w = w
        self.h = h

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        
        # init game state
        self.direction = RIGHT
        
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head, 
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)

        if self.food in self.snake:
            self._place_food()
    
    def _is_collision(self):
        # hits boundary
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True

        # hits itself
        if self.head in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, WHITE2, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, WHITE1, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render(str(self.score), True, WHITE1)
        text_rect = text.get_rect(center=(self.w / 2, 25))
        self.display.blit(text, text_rect)
        pygame.display.flip()
        
    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == RIGHT:
            x += BLOCK_SIZE
        elif direction == LEFT:
            x -= BLOCK_SIZE
        elif direction == DOWN:
            y += BLOCK_SIZE
        elif direction == UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)

    def reset(self):
        # init game state
        self.direction = RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head, 
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()

    def show_game_over_screen(self, score):
        self.display.fill(BLACK)
        
        game_over_text = font.render("GAME OVER", True, RED)
        game_over_rect = game_over_text.get_rect(center=(self.w / 2, self.h / 2 - 50))
        self.display.blit(game_over_text, game_over_rect)

        score_text = font.render("Score: " + str(score), True, WHITE1)
        score_rect = score_text.get_rect(center=(self.w / 2, self.h / 2))
        self.display.blit(score_text, score_rect)

        play_again_text = font.render("Play Again", True, BLACK)
        play_again_rect = play_again_text.get_rect(center=(self.w / 2, self.h / 2 + 50))
        button_rect = play_again_rect.inflate(20, 10) # padding around the text
        pygame.draw.rect(self.display, WHITE1, button_rect)
        self.display.blit(play_again_text, play_again_rect)

        pygame.display.flip()

        # wait for the user to click play again
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if play_again_rect.collidepoint(mouse_pos):
                        return

    def play(self):
        # user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = UP
                elif event.key == pygame.K_DOWN:
                    self.direction = DOWN
        
        # move
        self._move(self.direction) # update the head
        self.snake.insert(0, self.head)
        
        # check if game over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score
            
        # place new food or just move
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        
        # update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # return game over and score
        return game_over, self.score
            

if __name__ == "__main__":
    game = SnakeGame()
    
    # main game loop
    while True:
        game_over, score = game.play()
        
        if game_over == True:
            game.show_game_over_screen(score)
            game.reset()
        
    print("Final score:", score)
    
    pygame.quit()
