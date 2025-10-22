#Prompt for Claude 4.5 Sonnet inside Cursor: Write a simple script that generates 1000 random Fibonacci sequences, each of length 9. Each sequence starts with two random numbers between 1 and 20.

#!/usr/bin/env python3
"""
Fibonacci Sequence Generator

Generates 1000 random Fibonacci sequences, each of length 9.
Each sequence starts with two random numbers between 1 and 20.
"""

import random


def generate_fibonacci_sequence(start1, start2, length=9):
    """
    Generate a Fibonacci sequence starting with two given numbers.
    
    Args:
        start1: First number in the sequence
        start2: Second number in the sequence
        length: Total length of the sequence (default: 9)
        
    Returns:
        List containing the Fibonacci sequence
    """
    sequence = [start1, start2]
    
    for i in range(length - 2):
        next_num = sequence[-1] + sequence[-2]
        sequence.append(next_num)
    
    return sequence


def generate_random_fibonacci_sequences(count=1000, length=9, min_val=1, max_val=20):
    """
    Generate multiple random Fibonacci sequences.
    
    Args:
        count: Number of sequences to generate (default: 1000)
        length: Length of each sequence (default: 9)
        min_val: Minimum value for random starting numbers (default: 1)
        max_val: Maximum value for random starting numbers (default: 20)
        
    Returns:
        List of Fibonacci sequences
    """
    sequences = []
    
    for _ in range(count):
        start1 = random.randint(min_val, max_val)
        start2 = random.randint(min_val, max_val)
        sequence = generate_fibonacci_sequence(start1, start2, length)
        sequences.append(sequence)
    
    return sequences


def main():
    """Main function to generate and display Fibonacci sequences."""
    # Generate 1000 random Fibonacci sequences
    sequences = generate_random_fibonacci_sequences(count=1000, length=9)
    
    # Print first 10 sequences as examples
    print("Generated 1000 random Fibonacci sequences (length 9)")
    print("\nFirst 10 sequences:")
    for i, seq in enumerate(sequences[:10], 1):
        print(f"{i:3d}: {seq}")
    
    # Save all sequences to a file
    output_file = "data/fib_all.txt"
    with open(output_file, 'w') as f:
        f.write("# 1000 Random Fibonacci Sequences (length 9)\n")
        f.write("# Each sequence starts with two random numbers between 1 and 20\n\n")
        for i, seq in enumerate(sequences, 1):
            seq_str = ','.join(map(str, seq))
            f.write(f"{seq_str}\n")
    
    print(f"\nAll sequences saved to: {output_file}")
    print(f"Total sequences generated: {len(sequences)}")


if __name__ == "__main__":
    main()
