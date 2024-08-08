import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score # type: ignore
from model import Linear_QNet

INPUT_SIZE = 12
HIDDEN_SIZE = 255
OUTPUT_SIZE = 3

# Assuming you have a dataset class defined
# from your_dataset import YourDataset

# Load the model from the file
model_dict = torch.load('model/model.pth')
model = Linear_QNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
model.load_state_dict(model_dict)

# Set the model to evaluation mode
model.eval()

# Assuming you have a dataset and a DataLoader
dataset = YourDataset('path/to/your/evaluation/data')
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Example evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# Perform evaluation on your data
accuracy = evaluate_model(model, dataloader)
print(f'Accuracy: {accuracy:.4f}')

# Note: Uncomment the dataset and dataloader lines and replace 'YourDataset' with your actual dataset class.