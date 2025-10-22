#Prompt for Claude 4.5 Sonnet inside Cursor: Write a very simple script that loads the models/fib_model.pth file and tests it on a single sequence [1, 1, 2, 3, 5, 8, 13, 21]

#Learned from interactions with Cursor's Claude 4.5: 
# models/fib_model.pth contains:
# ✅ fc1.weight = [array of 512 numbers]
# ✅ fc1.bias = [array of 64 numbers]
# ✅ fc2.weight = [array of 2048 numbers]
# ✅ fc2.bias = [array of 32 numbers]
# ✅ fc3.weight = [array of 32 numbers]
# ✅ fc3.bias = [1 number]

# ❌ Does NOT contain:
#    - How many layers?
#    - What layer types?
#    - What activation functions?
#    - How are they connected?

import torch
import torch.nn as nn
import random

# Define the same model architecture
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

# Load the model
model = MySequenceNet(input_size=8)
model.load_state_dict(torch.load('models/fib_model.pth'))
model.eval()

# Test sequence
sequence = [10, 10, 0, 0, 0, 0, 13, 21]
input_tensor = torch.FloatTensor(sequence)

# Make prediction
with torch.no_grad():
    prediction = model(input_tensor).item()
    rounded_prediction = round(prediction)
    
print(f"Input sequence: {sequence}")
print(f"Output prediction: {prediction}")
print(f"Predicted next number: {rounded_prediction}")
print(f"Actual next number: 34")

# Choose a random sequence from fib_test.txt
with open('data/fib_test.txt', 'r') as f:
    lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
random_line = random.choice(lines)
fib_test_sequence = [int(x) for x in random_line.split(',')]
fib_test_input = fib_test_sequence[:8]  # First 8 numbers as input
fib_test_actual = fib_test_sequence[8]  # 9th number as target
fib_test_tensor = torch.FloatTensor(fib_test_input)

# Make prediction
with torch.no_grad():
    prediction = model(fib_test_tensor).item()
    rounded_prediction = round(prediction)

print(f"\nRandom sequence from fib_test: {fib_test_input}")
print(f"Output prediction: {prediction:.2f}")
print(f"Predicted next number: {rounded_prediction}")
print(f"Actual next number: {fib_test_actual}")