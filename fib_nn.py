#Prompt For Claude Sonnet 4.5 within VSCode's Github Copilot:  Create a very basic version of a feedforward neural network that predicts the next number in a sequence.  Please use pytorch.  Keep it simple--no frills 

#!/usr/bin/env python3
"""
Simple Feedforward Neural Network for Sequence Prediction
Uses PyTorch to predict the next number in a Fibonacci sequence.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Simple 3-layer (2 hidden layers)feedforward neural network
# Input: [8 numbers]
#    ↓
# [Linear: 8 → 64]  ← fc1 (512 weights + 64 biases = 576 parameters)
#    ↓
# [ReLU activation]
#    ↓
# [Linear: 64 → 32]  ← fc2 (2048 weights + 32 biases = 2080 parameters)
#    ↓
# [ReLU activation]
#    ↓
# [Linear: 32 → 1]   ← fc3 (32 weights + 1 bias = 33 parameters)
#    ↓
# Output: [1 number]
#Total: 2,689 trainable parameters

class MySequenceNet(nn.Module):
    def __init__(self, input_size=8):
        super(MySequenceNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Dataset class
class FibDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    numbers = [int(x) for x in line.split(',')]
                    if len(numbers) >= 9:
                        # First 8 numbers as input, 9th as target
                        self.data.append((numbers[:8], numbers[8]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.FloatTensor(x), torch.FloatTensor([y])

# Training function
def train_model(model, train_loader, epochs, lr=0.001):
    criterion = nn.MSELoss(reduction='sum')  # Sum instead of mean
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        total_samples = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_samples += len(targets)
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / total_samples
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

# Testing function
def test_model(model, test_loader):
    model.eval()
    total_error = 0
    total_loss = 0
    total_items = 0
    i=0
    with torch.no_grad(): #turns off internal pytorch gradient tracking to be faster during testing
        for inputs, targets in test_loader:
            i+=1
            outputs = model(inputs)
            loss = ((outputs - targets) ** 2).sum()            
            error = torch.abs(outputs - targets).sum()
            num_targtes = len(targets)
            #print(f"Average error on {i}th iteration: {error/num_targtes}")
            total_error += error
            total_loss += loss
            total_items += len(targets)
    
    print(f"Length of test loader: {len(test_loader)}")
    avg_error = total_error / total_items
    print(f'Average Test Error: {avg_error:.2f}')
    avg_loss = total_loss / total_items
    print(f'Average Test Loss: {avg_loss:.2f}')
    return avg_error

def main():
    # Load data
    train_dataset = FibDataset('data/fib_train.txt')
    test_dataset = FibDataset('data/fib_test.txt')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    model = MySequenceNet(input_size=8)
    
    # Train
    print("\nTraining...")
    train_model(model, train_loader, epochs=500, lr=0.001)
    
    # Test
    print("\nTesting...")
    test_model(model, test_loader)
    
    # Save model
    torch.save(model.state_dict(), 'models/fib_model.pth')
    print("\nModel saved to models/fib_model.pth")

if __name__ == '__main__':
    main()

