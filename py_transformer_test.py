import math
import torch
from typing import List, Tuple
from py_transformer_train import NextNumberTransformer, read_sequences, compute_mean_std, predict_next

if __name__ == "__main__":
    # sequences = read_sequences("data/fib_test.txt")
    # if not sequences:
    #     raise SystemExit("No sequences found in data/fib_test.txt")

    # mean, std = compute_mean_std(sequences)
    # print(f"Mean: {mean}, Std: {std}")
    with open("data/fib_stats.txt", "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    mean = float(lines[0])
    std = float(lines[1])
    print(f"Mean: {mean}, Std: {std}")

    model = NextNumberTransformer(d_model=64, nhead=2, num_layers=2, dim_ff=128, dropout=0.1)
    state = torch.load("models/py_transformer_model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    test_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34]
    pred = predict_next(model, test_sequence, mean, std)
    print(int(round(pred)))
