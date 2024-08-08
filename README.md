# Snake AI


# Set Up
Pytorch: https://pytorch.org/
Markdown Preview: Ctr + Shift + V


# Function Explain and Documentation (Organized this later)
## agent.py
### get_action() - returns the action the agent should take
In reinforcement learning, it is common to have a tradeoff between exploration and exploitation. Exploration refers to taking random actions to discover new states and learn more about the environment, while exploitation refers to taking actions that are predicted to have the highest reward based on previous knowledge.

+ In this case, the if statement checks if a randomly generated number is less than the exploration rate (self.epsilon). If it is, a random action is chosen by setting one element of the final_move list to 1. This random action allows the agent to explore the environment and gather more information about different states and their corresponding rewards.

+ On the other hand, if the randomly generated number is greater than or equal to the exploration rate, the else statement is executed. In this case, the agent uses its learned knowledge to predict the action with the highest expected reward. The state is converted into a tensor format and passed to the predict method of the self.model object. The predicted action is then determined by finding the index of the maximum value in the prediction tensor using torch.argmax. The corresponding element in the final_move list is set to 1, indicating the chosen action.


