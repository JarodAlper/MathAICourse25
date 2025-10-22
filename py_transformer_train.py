#Prompt in Cursor using GPT5: Using the Pytorch transformer torch.nn.Transformer as documented here @https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer , please write a very simply Python script that trains a transformer to predict the next number using the dataset data/fib_test.txt, and then runs the model on the sequence [1, 1, 2, 3, 5, 8, 13, 21].  Keep it simple.


import math
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple


def read_sequences(path: str) -> List[List[float]]:
    sequences: List[List[float]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) < 2:
                continue
            sequences.append([float(p) for p in parts])
    print(f"Sequence 1: {sequences[1]}")
    print(f"Total sequences: {len(sequences)}")
    return sequences


def compute_mean_std(sequences: List[List[float]]) -> Tuple[float, float]:
    all_vals = [v for seq in sequences for v in seq]
    mean = sum(all_vals) / len(all_vals)
    var = sum((v - mean) ** 2 for v in all_vals) / max(len(all_vals) - 1, 1)
    std = math.sqrt(var) if var > 0 else 1.0
    return mean, std


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 1024) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    # (sz, sz) with -inf above diagonal
    mask = torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)
    return mask


class NextNumberTransformer(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 2, num_layers: int = 2, dim_ff: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
        )
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # src, tgt: (seq_len, batch, 1)
        src_emb = self.pos_enc(self.input_proj(src))  # (S, B, d_model)
        tgt_emb = self.pos_enc(self.input_proj(tgt))  # (T, B, d_model)
        tgt_mask = generate_square_subsequent_mask(tgt_emb.size(0)).to(tgt_emb.device)
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)  # (T, B, d_model)
        return self.output_proj(out)  # (T, B, 1)


def build_batch(sequences: List[List[float]], mean: float, std: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Assume all sequences are equal length; if not, you would need padding.
    length = min(len(seq) for seq in sequences)
    # Use the first (length-1) as inputs, predict the following (length-1)
    src_list = []
    tgt_in_list = []
    tgt_out_list = []
    for seq in sequences:
        seq = seq[:length]
        norm = [(x - mean) / std for x in seq]
        src_seq = norm[:-1]
        tgt_in_seq = norm[:-1]
        tgt_out_seq = norm[1:]
        src_list.append(src_seq)
        tgt_in_list.append(tgt_in_seq)
        tgt_out_list.append(tgt_out_seq)

    # Shapes: (S, B, 1)
    src = torch.tensor(src_list, dtype=torch.float32).transpose(0, 1).unsqueeze(-1)
    tgt_in = torch.tensor(tgt_in_list, dtype=torch.float32).transpose(0, 1).unsqueeze(-1)
    tgt_out = torch.tensor(tgt_out_list, dtype=torch.float32).transpose(0, 1).unsqueeze(-1)
    return src, tgt_in, tgt_out


def train_model(model: nn.Module, src: torch.Tensor, tgt_in: torch.Tensor, tgt_out: torch.Tensor, epochs: int = 200, lr: float = 1e-3) -> None:
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(src, tgt_in)
        loss = criterion(pred, tgt_out)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}: loss={loss.item():.6f}")


def predict_next(model: nn.Module, sequence: List[float], mean: float, std: float) -> float:
    model.eval()
    with torch.no_grad():
        # Condition on the entire sequence and ask for next step
        norm = [(x - mean) / std for x in sequence]
        src = torch.tensor(norm, dtype=torch.float32).unsqueeze(1).unsqueeze(-1)  # (L, 1, 1)
        tgt_in = torch.tensor(norm, dtype=torch.float32).unsqueeze(1).unsqueeze(-1)  # (L, 1, 1)
        out = model(src, tgt_in)  # (L, 1, 1)
        next_norm = out[-1, 0, 0].item()
        return next_norm * std + mean


if __name__ == "__main__":
    sequences = read_sequences("/Users/jarod/gitwork/math-ai-course/data/fib_test.txt")
    if not sequences:
        raise SystemExit("No sequences found in data/fib_test.txt")

    mean, std = compute_mean_std(sequences)

    model = NextNumberTransformer(d_model=64, nhead=2, num_layers=2, dim_ff=128, dropout=0.1)

    src, tgt_in, tgt_out = build_batch(sequences, mean, std)

    train_model(model, src, tgt_in, tgt_out, epochs=500, lr=1e-3)

    torch.save(model.state_dict(), "py_transformer_model.pth")

    test_sequence = [1, 1, 2, 3, 5, 8, 13, 21]
    pred = predict_next(model, test_sequence, mean, std)
    print(f"Test sequence: {test_sequence} Predicted next number: {int(round(pred))}")

    test_sequence2 = [1, 1, 2, 3, 5, 8, 13]
    pred2 = predict_next(model, test_sequence2, mean, std)
    print(f"Test sequence: {test_sequence2} Predicted next number: {int(round(pred2))}")