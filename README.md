# Files
**agent.py:** Contains the Agent class, which implements the Q-learning algorithm to train the agent to play the game.

**model/model.pth:** Contains the trained model weights after training the agent.

**evaluation:** Contains the evaluation function, which evaluates the performance of the agent by running multiple episodes and calculating the average reward.

**helper.py:** Contains helper functions to preprocess the game state and convert it into a format 
that can be used by the neural network.

**model.py:** Contains the DQN class, which defines the neural network architecture used to approximate the Q-function.

**snake.py:** Contains the SnakeEnv class, which defines the environment for the game and implements the step function to interact with the agent.

# Set Up
Install Pytorch: https://pytorch.org/

Install Requirement: pip install -r requirements.txt
Run agent.py file

Markdown Preview: Ctr + Shift + V
 

# Documentation (On Development)
## Functions
### agent.py
`get_action()` - returns the action the agent should take

In reinforcement learning, it is common to have a tradeoff between exploration and exploitation. Exploration refers to taking random actions to discover new states and learn more about the environment, while exploitation refers to taking actions that are predicted to have the highest reward based on previous knowledge.

+ In this case, the if statement checks if a randomly generated number is less than the exploration rate (self.epsilon). If it is, a random action is chosen by setting one element of the final_move list to 1. This random action allows the agent to explore the environment and gather more information about different states and their corresponding rewards.

+ On the other hand, if the randomly generated number is greater than or equal to the exploration rate, the else statement is executed. In this case, the agent uses its learned knowledge to predict the action with the highest expected reward. The state is converted into a tensor format and passed to the predict method of the self.model object. The predicted action is then determined by finding the index of the maximum value in the prediction tensor using torch.argmax. The corresponding element in the final_move list is set to 1, indicating the chosen action.


## Algorithm
### Q-Learning
### Deep Q-Learning


# TO-DO:
+ Extract Feature for Evolution Algorithm
+ Implement Genetic Algorithm
+ Create a file for Natural Selection 
 