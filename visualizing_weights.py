# Prompt for Claude 4.5 Sonnet: Generate a very simple script that inspects models/fib_model.pth and prints out the weight matrices and bias vectors.  Keep it simple

import torch

# Load the saved model parameters
model_state = torch.load('models/fib_model.pth')

print("=" * 60)
print("Model Parameters in models/fib_model.pth")
print("=" * 60)

for param_name, param_tensor in model_state.items():
    print(f"\n{param_name}")
    print(f"Shape: {param_tensor.shape}")
    print(f"Values:\n{param_tensor}")
    print("-" * 60)

