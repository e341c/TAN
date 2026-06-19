import numpy as np
import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for i, q in enumerate(self.quantiles):
            error = target - preds[:, i]
            losses.append(torch.max((q - 1) * error, q * error).unsqueeze(1))
        return torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))


class QuantileLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 64,
        quantiles: list[float] | None = None,
    ):
        super().__init__()
        quantiles = quantiles or [0.1, 0.5, 0.9]
        self.quantiles = quantiles
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, len(quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def train_quantile_lstm(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
) -> tuple[nn.Module, dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    model = QuantileLSTM(
        input_size=X.shape[2],
        hidden_size=cfg.get("hidden_size", 50),
        quantiles=cfg.get("quantiles", [0.1, 0.5, 0.9]),
    ).to(device)

    criterion = QuantileLoss(model.quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("learning_rate", 1e-3))

    batch_size = cfg.get("batch_size", 64)
    epochs = cfg.get("epochs", 100)
    n_samples = X.shape[0]

    for epoch in range(epochs):
        permutation = np.random.permutation(n_samples)
        epoch_loss = 0.0
        for i in range(0, n_samples, batch_size):
            idx = permutation[i : i + batch_size]
            batch_X = X_tensor[idx]
            batch_y = y_tensor[idx]
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n_samples
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.6f}")

    return model, {"final_loss": epoch_loss}
