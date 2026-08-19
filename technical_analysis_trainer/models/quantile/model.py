import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = list(quantiles)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        q = torch.tensor(self.quantiles, dtype=preds.dtype, device=preds.device)
        error = target.unsqueeze(1) - preds
        return torch.mean(torch.sum(torch.maximum((q - 1) * error, q * error), dim=1))


class QuantileLSTM(nn.Module):
    """
    LSTM over a window of stationary features → quantiles of the
    volatility-normalised forward return.

    The head emits a base level plus positive increments, so the outputs are
    monotonically non-decreasing by construction: q10 <= q50 <= q90 can never
    cross, which an unconstrained Linear head allows and which produces
    nonsensical intervals on the exact bars you care about most.
    """

    def __init__(
        self,
        input_size: int = 21,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout_lstm: float = 0.2,
        dropout_fc: float = 0.1,
        quantiles: list[float] | None = None,
    ):
        super().__init__()
        quantiles = quantiles or [0.1, 0.5, 0.9]
        self.quantiles = quantiles
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_lstm if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_fc)
        self.fc = nn.Linear(hidden_size, len(quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        raw = self.fc(self.dropout(lstm_out[:, -1, :]))
        base = raw[:, :1]
        deltas = F.softplus(raw[:, 1:])
        return torch.cumsum(torch.cat([base, deltas], dim=1), dim=1)


def _enforce_monotone(out: torch.Tensor) -> torch.Tensor:
    """
    Project outputs onto the non-decreasing cone using only relu/cumsum/cat,
    all of which export cleanly to ONNX (torch.cummax does not).
    """
    base = out[:, :1]
    diffs = torch.relu(out[:, 1:] - out[:, :-1])
    return torch.cat([base, base + torch.cumsum(diffs, dim=1)], dim=1)


class QuantileEnsemble(nn.Module):
    """
    Seed ensemble with conformal offsets baked in, exportable as one ONNX graph.

    Two fixes in one wrapper:

      * averaging over seeds removes the run-to-run variance that otherwise
        tempts you into picking the luckiest draw;
      * `offsets` are split-conformal corrections fitted on a calibration split,
        so empirical coverage matches nominal on unseen data rather than only
        approximately.

    Offsets live in volatility-normalised space, so they scale correctly when
    inference multiplies the output by the current sigma.
    """

    def __init__(self, members: list[nn.Module], offsets=None):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.quantiles = getattr(members[0], "quantiles", [0.1, 0.5, 0.9])
        n_q = len(self.quantiles)
        if offsets is None:
            offsets = np.zeros(n_q, dtype=np.float32)
        self.register_buffer(
            "offsets", torch.as_tensor(np.asarray(offsets), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack([m(x) for m in self.members], dim=0)
        return _enforce_monotone(stacked.mean(dim=0) + self.offsets)


def train_quantile_lstm(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    cfg: dict | None = None,
) -> tuple[nn.Module, dict]:
    cfg = cfg or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)

    model = QuantileLSTM(
        input_size=X.shape[2],
        hidden_size=cfg.get("hidden_size", 64),
        num_layers=cfg.get("num_layers", 2),
        dropout_lstm=cfg.get("dropout_lstm", 0.2),
        dropout_fc=cfg.get("dropout_fc", 0.1),
        quantiles=cfg.get("quantiles", [0.1, 0.5, 0.9]),
    ).to(device)

    criterion = QuantileLoss(model.quantiles)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("learning_rate", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    if X_val is not None and y_val is not None and len(X_val) > 0:
        X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_v = torch.tensor(y_val, dtype=torch.float32).to(device)
    else:
        X_v = y_v = None

    batch_size = cfg.get("batch_size", 64)
    epochs = cfg.get("epochs", 200)
    patience = cfg.get("patience", 20)
    n_samples = X.shape[0]

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    no_improve = 0
    stopped_early = False

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0.0
        for i in range(0, n_samples, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_t[idx]), y_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n_samples
        history["train_loss"].append(epoch_loss)

        val_info = ""
        if X_v is not None:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_v), y_v).item()
            history["val_loss"].append(val_loss)
            val_info = f" | val_loss: {val_loss:.6f}"

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.6f}{val_info}")

        if X_v is not None and no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1} (best val_loss={best_val:.6f})")
            stopped_early = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_losses = history["val_loss"]
    return model, {
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "best_val_loss": float(min(val_losses)) if val_losses else None,
        "best_epoch": int(1 + np.argmin(val_losses)) if val_losses else 0,
        "stopped_early": stopped_early,
        "history": history,
    }
