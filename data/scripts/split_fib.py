#Prompt for Claude 4.5 Sonnet inside Cursor: Using the scikit-learn library, write a very, very simple script that splits the data/fib_all dataset into 75% training data and 25% test data

#!/usr/bin/env python3
"""Simple script to split Fibonacci dataset using scikit-learn"""

from sklearn.model_selection import train_test_split

# Read all lines from the file
with open('data/fib_all.txt', 'r') as f:
    lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Split into 75% training and 25% test
train_data, test_data = train_test_split(lines, test_size=0.25, random_state=42)

# Save training data
with open('data/fib_train.txt', 'w') as f:
    f.write('# Fibonacci Training Dataset (75%)\n')
    for line in train_data:
        f.write(line + '\n')

# Save test data
with open('data/fib_test.txt', 'w') as f:
    f.write('# Fibonacci Test Dataset (25%)\n')
    for line in test_data:
        f.write(line + '\n')

print(f"Total sequences: {len(lines)}")
print(f"Training: {len(train_data)} ({len(train_data)/len(lines)*100:.1f}%)")
print(f"Test: {len(test_data)} ({len(test_data)/len(lines)*100:.1f}%)")
print("Saved to data/fib_train.txt and data/fib_test.txt")
