# trend_model.py
import torch
import torch.nn as nn
import numpy as np


class TrendLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size=64, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1, :])
        return logits


def train_trend_lstm(X: np.ndarray, y: np.ndarray, cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TrendLSTM(
        input_size=X.shape[2],
        hidden_size=cfg.get("hidden_size", 64),
        num_classes=3,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 1e-3),
    )
    criterion = nn.CrossEntropyLoss()

    X_t = torch.tensor(X).to(device)
    y_t = torch.tensor(y).to(device)

    epochs = cfg.get("epochs", 100)
    batch_size = cfg.get("batch_size", 64)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X_t))
        total_loss = 0.0

        for i in range(0, len(X_t), batch_size):
            idx = perm[i : i + batch_size]
            xb = X_t[idx]
            yb = y_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] loss = {total_loss:.4f}")

    return model, {"classes": ["DOWN", "FLAT", "UP"]}


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray):
    model.eval()
    logits = model(torch.tensor(X))
    probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()
