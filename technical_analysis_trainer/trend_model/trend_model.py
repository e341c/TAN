import torch
import torch.nn as nn
import numpy as np


# Мы исчерпали то что можно выжать из текущего подхода. Проблема фундаментальная — LSTM пытается найти устойчивый паттерн в данных где его нет в явном виде на 1H таймфрейме с текущими фичами.
# Есть два честных пути вперёд:
# Путь 1 — сменить архитектуру на Transformer. Temporal Fusion Transformer или простой Transformer encoder справляются с финансовыми временными рядами лучше LSTM за счёт механизма attention — он сам выбирает какие моменты в окне важны. Это большой рефакторинг но реальный шаг вперёд.
# Путь 2 — переосмыслить задачу. Вместо предсказания направления следующей свечи предсказывать вероятность того что цена достигнет +X% раньше чем -X% в следующие N баров. Это более стабильный сигнал и так работают многие реальные торговые системы.
class TrendLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_layers: int = 1,
        hidden_size: int = 64,
        num_classes: int = 2,
    ):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(input_size)  # нормализуем входные фичи по батчу
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2 if num_layers > 1 else 0.0,  # dropout только при >1 слое
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        # BatchNorm1d ожидает (batch, features, seq_len)
        x = self.bn_input(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1, :])
        return logits


def train_trend_lstm(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    cfg: dict = {},
) -> tuple[nn.Module, dict]:
    """
    Обучает TrendLSTM на бинарную классификацию (0=DOWN, 1=UP).

    Args:
        X, y         — train-данные
        X_val, y_val — val-данные для early stopping (опционально)
        cfg          — гиперпараметры из config.yaml
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TrendLSTM(
        input_size=X.shape[2],
        num_layers=cfg.get("num_layers", 1),
        hidden_size=cfg.get("hidden_size", 64),
        num_classes=2,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 3e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    criterion = (
        nn.CrossEntropyLoss()
    )  # без class_weight — классы сбалансированы (48/52)

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.long).to(device)

    if X_val is not None and y_val is not None:
        X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_v = torch.tensor(y_val, dtype=torch.long).to(device)
    else:
        X_v = y_v = None

    epochs = cfg.get("epochs", 100)
    batch_size = cfg.get("batch_size", 64)
    patience = cfg.get("patience", 30)
    num_batches = max(1, len(X_t) // batch_size)

    history = {"train_loss": [], "val_loss": []}

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    stopped_early = False

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_t))
        total_loss = 0.0

        for i in range(0, len(X_t), batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_t[idx], y_t[idx]

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        history["train_loss"].append(avg_loss)

        val_info = ""
        if X_v is not None:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_v), y_v).item()
            history["val_loss"].append(val_loss)
            val_info = f"  val_loss = {val_loss:.4f}"

            if val_loss < best_val_loss - 1e-3:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

        print(f"[Epoch {epoch+1:>3}/{epochs}] loss = {avg_loss:.4f}{val_info}")

        if X_v is not None and no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}  (best val_loss={best_val_loss:.4f})"
            )
            stopped_early = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "classes": ["DOWN", "UP"],
        "history": history,
        "stopped_early": stopped_early,
    }


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Возвращает вероятности классов [DOWN, UP] формы (N, 2)."""
    model.eval()
    logits = model(torch.tensor(X, dtype=torch.float32))
    return torch.softmax(logits, dim=1).cpu().numpy()
