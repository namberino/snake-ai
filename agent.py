import torch
import random
import numpy as np
from collections import deque  # use to store data

import snake
from snake import SnakeAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    """
    The Agent class represents an agent for the SnakeAI Game.
    
    Attributes:
        n_games (int): The number of games played by the agent.
        epsilon (float): The randomness factor for exploration.
        gamma (float): The discount rate for future rewards.
        memory (deque): A deque to store game data.
        model (object): The model used for prediction.
        trainer (object): The trainer used for training the model.
    Methods:
        get_state(game): Returns the current state of the game.
        remember(state, action, reward, next_state, done): Stores game data in memory.
        train_long_memory(): Trains the model using long-term memory.
        train_short_memory(state, action, reward, next_state, done): Trains the model using short-term memory.
        get_action(state): Returns the action to take based on the current state.
    """
        
          
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  #randomness?
        self.gamma = 0  # discount rate?
        self.memory = deque(maxlen=MAX_MEMORY)  # save data to deque
        self.model = None
        self.trainer = None

    def get_state(self, game):
        """
        Returns the state of the game for the agent in 0 and 1 format.
        
        Parameters:
            game (Game): The game object representing the current state of the game.
        Returns:
            state (numpy.ndarray): An array representing the state of the game. The array contains the following elements:
                Danger Direction: True if there is a danger straight/right/left/below ahead in the current direction, False otherwise
                Food Location: True if the food is to the left/right/above/below of the snake's head, False otherwise.
        """

        head = game.snake[0]
        point_l = Point(head.x + 20, head.y)
        point_r = Point(head.x - 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_r = game.direction == Direction.RIGHT
        dir_l = game.direction == Direction.LEFT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN


        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            # Ex: True if there is a danger to the right in the current direction, False otherwise
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Danger down
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food Location (start from head location)
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y > game.head.y,  # food up
            game.food.y < game.head.y,  # food down
        ]

        return np.array(state, dtype=int)  # Turn State (True or False boolean) into 0 and 1


    def remember(self, state, action, reward, next_state, done):
        # append a list of tuple
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached


    """
    Train_long_memory & Train short memory

    Args:
        state (np.array): The current state of the game.
        action (int): The action taken.
        reward (float): The reward received.
        next_state (np.array): The next state of the game.
        done (bool): Indicates if the game is done.

    Returns:
        the action to take based on the current state.
    """
    def train_long_memory(self):
        """
        This method is responsible for training the agent using the samples stored in the long-term memory.
        If the number of samples in the memory is greater than the batch size, a mini-batch of size BATCH_SIZE is randomly selected.
        Otherwise, all the samples in the memory are used.
        """
        if len(self.memory) > BATCH_SIZE:  # if have more than BATCH_SIZE samples
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)  # Extract every samples existed
        self.trainer.train_step(states, actions, rewards, next_states, dones)  # Train with every samples existed


    def train_short_memory(self, state, action, reward, next_state, done):  # 1 step    
        self.trainer.train_step(state, action, reward, next_state, done)



    def get_action(self, state):
        """
        Returns a one-hot encoded list representing the action to take based on the given state.
        Parameters:
        - state: The current state of the game.
        Returns:
        - final_move: A one-hot encoded list [0, 0, 0] where only one element is 1, indicating the direction to move.
        """
        # random moves: tradeoff exploration  / exploration
        self.epsilon = 69 - self.n_games  # epsilon get smaller as the game progress

        # Return one-hot encoded list [0, 0, 0] where only one element is 1 indicating the direction.
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # get raw values and convert max value into 1 other values 0.
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model.predict(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    """
    Trains the snake AI agent by playing the snake game.
    This function iteratively plays the snake game and trains the agent's neural network model.
    It keeps track of the scores and records, and updates the agent's memory for training.
    After each game, it resets the snake game and trains the agent's long-term memory.
    """
    # code implementation
    ...
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    snake = SnakeAI()

    while True:
        # get old state
        state_old = agent.get_state(snake)

        # get move
        final_move = agent.get_action(state_old)

        # perform move andd get new state
        reward, done, score = snake.play(final_move)
        state_new = agent.get_state(snake)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory. replay memory
            snake.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                reward = score
                # agent.mode.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            #TODO: plot


if __name__ == '__main__':
    train()
