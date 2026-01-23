import torch
import torch.nn as nn
import numpy as np


class TechnicalRRN(nn.Module):
    def __init__(self, hidden_size=50, output_size=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,  # feature_min=None, feature_max=None
) -> tuple[nn.Module, dict]:
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)  # Add feature dimension
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Model parameters
    hidden_size = cfg.get("hidden_size", 50)
    learning_rate = cfg.get("learning_rate", 0.001)
    epochs = cfg.get("epochs", 100)

    # Initialize model, loss function, and optimizer
    model = TechnicalRRN(hidden_size=hidden_size, output_size=5)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    train_metrics = {"final_loss": loss.item()}  # type: ignore

    return model, train_metrics


# ======================
# Quantile Loss (Pinball)
# ======================
class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        preds: (batch, n_quantiles)
        target: (batch,)
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            error = target - preds[:, i]
            losses.append(torch.max((q - 1) * error, q * error).unsqueeze(1))
        return torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))


# ======================
# LSTM Model for Quantiles
# ======================
# 30–60 свечей → hidden_size = 32–64
class QuantileLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 64,
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.quantiles = quantiles

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, len(quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_size)
        out = self.fc(lstm_out[:, -1, :])  # take last hidden state
        return out


# ❌ Предсказывать сразу Open/High/Low/Close одной моделью
# ❌ Масштабировать y
# ❌ Смешивать будущее в окне (data leakage)
# ❌ Интерпретировать квантиль как вероятность роста

# Модель будет лучше работать с returns
# Qτ(log_return_31)


def train_quantile_lstm(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
) -> tuple[nn.Module, dict]:
    """
    X: (n_samples, seq_len, n_features) -- уже scaled
    y: (n_samples,) -- future log-return (float)
    cfg: {
        "hidden_size": int,
        "learning_rate": float,
        "epochs": int,
        "quantiles": list[float],
        "batch_size": int
    }
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    # Model
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

    train_metrics = {"final_loss": epoch_loss}  # type: ignore

    return model, train_metrics
