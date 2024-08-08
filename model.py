import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import os


#TODO: Learn How to Modify Q-Learning 
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    #? Save Training Result to model.pth (pytorch file format) 
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()  # mean square error
        self.training_log = []  # List to store training results

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x) append 1D in the beginning
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)  # define 1 value tuple

        # 1: predicted Q values with current state
        pred = self.model(state)
        target = pred.clone()

        for index in range(len(done)):
            Q_new = reward[index]
            if not done[index]:
                Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))

            target[index][torch.argmax(action[index]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done  
        # pred.clone() -> get current state for prediction
        # preds[argmax(action)] = Q_new -> index of 1 in action (one-hot encoded) get send to Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        
                # Log the training results
        self.training_log.append({
            'state': state.numpy().tolist(),
            'action': action.numpy().tolist(),
            'reward': reward.numpy().tolist(),
            'next_state': next_state.numpy().tolist(),
            'done': done,
            'loss': loss.item()
        })

    def save_training_log(self, file_path):
        df = pd.DataFrame(self.training_log)
        df.to_csv(file_path, index=False)

# Example usage
if __name__ == "__main__":    
    INPUT_SIZE = 3
    HIDDEN_SIZE = 1
    OUTPUT_SIZE = 3

    model = Linear_QNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    trainer = QTrainer(model, lr=0.001, gamma=0.9)
    
    # Example training step
    trainer.train_step([1, 2, 3], [0], 1, [4, 5, 6], False)
    
    # Save the training log to a CSV file
    trainer.save_training_log('training_log.csv')