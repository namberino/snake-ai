from turtle import Screen
from snake_brain import snake
import time
from food import Food
from scoreBoard import ScoreBoard


#? Configuration
snake_speed = 0.065
SCREEN_WIDTH = 700
SCREEN_HEIGHT = 700


# TODO-5: Create a scoreboard
score_board = ScoreBoard()

# TODO-1: Setup Screen
screen = Screen()
screen.setup(width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
screen.bgcolor('crimson')
screen.title('Second Snake Game')
screen.tracer(0)  # smoother animation. [action per frame]

# TODO-2: Create Snake body
Snake = snake(3, 'white')
apple = Food()

screen.listen()  # listen before onkey()
screen.onkey(Snake.up, 'Up')
screen.onkey(Snake.down, 'Down')
screen.onkey(Snake.turn_right, 'Right')
screen.onkey(Snake.turn_left, 'Left')

# TODO-3: Control the Snake
is_alive = True
while is_alive:
	screen.update()
	time.sleep(snake_speed)  # delay time between block. refresh rate
	Snake.move()

	# TODO-4: Detect collision with food (head coords vs food coords)
	if Snake.head.distance(apple) < 15:
		print('nom nom nom')
		apple.refresh()
		Snake.extend()
		score_board.increase_score()

	collision_point_x = SCREEN_WIDTH/2 - 10
	collision_point_y = SCREEN_HEIGHT/2 - 10

	# TODO-6: Detect wall collision (head coords ~ wall coords)
	if Snake.head.xcor() > collision_point_x or Snake.head.xcor() < -collision_point_x or Snake.head.ycor() > collision_point_y or Snake.head.ycor() < -collision_point_y:
		is_alive = False
		score_board.lose()

	# TODO-7: Detect collision with tail
	for block in Snake.body[1:]:  # all block except the head
		if Snake.head.distance(block) < 15:
			is_alive = False
			score_board.lose()

# TODO-8: Replay ability (by clicking the screen)
screen.exitonclick()

