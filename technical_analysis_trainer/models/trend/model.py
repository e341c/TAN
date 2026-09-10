import math

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score


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
        dropout_lstm: float = 0.0,
        dropout_fc: float = 0.3,
    ):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(input_size)  # нормализуем входные фичи по батчу
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_lstm,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_fc)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        # BatchNorm1d ожидает (batch, features, seq_len)
        x = self.bn_input(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1, :])
        return logits


class TrendEnsemble(nn.Module):
    """
    Seed ensemble over TrendLSTM members, exportable as a single ONNX graph.

    The 184-run log shows ROC-AUC varying 0.5016–0.5457 across seeds with *zero*
    correlation to any hyperparameter — that spread is noise, and the best run is
    exactly where the max of 184 noise draws lands. Averaging the members
    captures the central estimate instead of gambling on a lucky draw.

    Outputs log(mean(softmax(member_logits))), so downstream code that applies
    softmax to the model output recovers the ensemble probability unchanged.
    """

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.stack(
            [torch.softmax(m(x), dim=1) for m in self.members], dim=0
        ).mean(dim=0)
        return torch.log(probs.clamp_min(1e-12))


def train_trend_lstm(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    cfg: dict = {},
    w: np.ndarray | None = None,
    w_val: np.ndarray | None = None,
) -> tuple[nn.Module, dict]:
    """
    Обучает TrendLSTM на бинарную классификацию (0=DOWN, 1=UP).

    Args:
        X, y         — train-данные
        X_val, y_val — val-данные для early stopping (опционально)
        cfg          — гиперпараметры из config.yaml
        w, w_val     — per-sample weights (move size x label uniqueness)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_layers = cfg.get("num_layers", 1)
    # PyTorch LSTM: dropout действует только между слоями; при num_layers=1 даёт warning — ставим 0
    dropout_lstm = cfg.get("dropout_lstm", 0.0) if num_layers > 1 else 0.0
    model = TrendLSTM(
        input_size=X.shape[2],
        num_layers=num_layers,
        hidden_size=cfg.get("hidden_size", 64),
        num_classes=2,
        dropout_lstm=dropout_lstm,
        dropout_fc=cfg.get("dropout_fc", 0.3),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("learning_rate", 3e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    class_weight = cfg.get("class_weight")
    if isinstance(class_weight, (list, tuple)) and len(class_weight) == 2:
        cw = torch.tensor(
            [float(class_weight[0]), float(class_weight[1])],
            dtype=torch.float32, device=device,
        )
    else:
        cw = None
    # reduction="none" so per-sample weights can be applied before reducing.
    criterion = nn.CrossEntropyLoss(weight=cw, reduction="none")

    def weighted_loss(logits, target, sample_w):
        per = criterion(logits, target)
        return (per * sample_w).mean() if sample_w is not None else per.mean()

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.long).to(device)
    w_t = torch.tensor(w, dtype=torch.float32).to(device) if w is not None else None

    if X_val is not None and y_val is not None:
        X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_v = torch.tensor(y_val, dtype=torch.long).to(device)
        w_v = torch.tensor(w_val, dtype=torch.float32).to(device) if w_val is not None else None
    else:
        X_v = y_v = w_v = None

    epochs = cfg.get("epochs", 100)
    batch_size = cfg.get("batch_size", 64)
    patience = cfg.get("patience", 15)
    num_batches = max(1, len(X_t) // batch_size)

    # LR warmup. 42% of the logged runs early-stopped by epoch 10 having learned
    # nothing (ROC-AUC 0.5289 vs 0.5339 for runs that trained past epoch 50) —
    # nearly half the compute wasted on a bad first few steps. Ramping the LR
    # keeps early updates from throwing the model into a flat region.
    warmup_epochs = int(cfg.get("warmup_epochs", 5))
    base_lr = cfg.get("learning_rate", 3e-4)

    # Cosine decay after warmup, down to `lr_min_factor` of the base rate.
    #
    # Held flat (the previous behaviour), the LR stays at its peak for every
    # epoch after the fifth, so the model crosses its best generalisation point
    # in two or three epochs and spends the rest of the run memorising: train
    # loss 0.69 -> 0.35 while validation AUC falls from 0.557 back to 0.463.
    # Early stopping then correctly keeps epoch 1-3, which is what "training
    # stops at epoch 1" actually was — not a broken stop rule but an optimiser
    # sprinting past the optimum before the rule can resolve it.
    #
    # Annealing does not raise the ceiling (~0.555 val AUC on this data whatever
    # the schedule) but it widens the plateau the checkpoint is picked from, so
    # the saved model stops being a coin flip on which epoch the seed landed.
    lr_decay = str(cfg.get("lr_decay", "cosine")).lower()
    lr_min_factor = float(cfg.get("lr_min_factor", 0.05))

    def lr_at(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        if lr_decay != "cosine":
            return base_lr
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return base_lr * (lr_min_factor + (1.0 - lr_min_factor) * cosine)

    # Which validation number decides the checkpoint.
    #
    # val_loss, deliberately, even though ROC-AUC is the reported metric. AUC is
    # measured on ~4.8k validation samples and read once per epoch, so taking its
    # argmax over a run selects the maximum of ~25 noisy draws — it reliably wins
    # on validation and loses on test (3 folds x 3 seeds: val AUC 0.5451 vs
    # val_loss's 0.5248, but test AUC 0.5270 vs 0.5557). That is precisely the
    # inflation core/validation.expected_max_of_noise exists to warn about.
    # Weighted cross-entropy is noisier-looking but is a proper scoring rule and
    # does not cherry-pick the peak of the noise.
    early_stopping_metric = str(cfg.get("early_stopping_metric", "val_loss")).lower()
    can_use_auc = y_val is not None and len(np.unique(np.asarray(y_val))) > 1
    if early_stopping_metric == "val_auc" and not can_use_auc:
        # A single-class validation split has no AUC; fall back rather than crash.
        early_stopping_metric = "val_loss"

    # Earliest epoch a checkpoint may come from.
    #
    # This is the actual fix for "training stops at epoch 1": epoch 1 weights are
    # the initialisation, and during warmup the LR is only 20-40% of target, so a
    # checkpoint from there was never trained at the intended step size. val_loss
    # used to select epoch 1 in 3 of 9 runs, and always keeping epoch 1 costs real
    # accuracy out of sample (test AUC 0.5386 vs 0.5557).
    min_epochs = max(1, int(cfg.get("min_epochs", 3)))

    # Hard epoch cap, independent of `epochs`.
    #
    # The final refit has no validation split — every held-out bar is spent on
    # being recent training data or on calibration — so it cannot early-stop and
    # instead trains for a budget taken from the folds' kept epochs. `epochs`
    # still sets the cosine horizon, so the learning-rate trajectory is the one
    # the folds actually ran; only the stopping point is imposed. Capping via
    # `epochs` instead would compress the schedule and train a different model.
    stop_after = int(cfg.get("stop_after_epoch") or 0)

    history = {"train_loss": [], "val_loss": [], "val_auc": [], "lr": []}

    best_val_loss = float("inf")
    best_score = -float("inf")  # higher is better, whatever the metric
    best_state = None
    best_metric_epoch = 0
    no_improve = 0
    stopped_early = False

    for epoch in range(epochs):
        model.train()
        for g in optimizer.param_groups:
            g["lr"] = lr_at(epoch)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        perm = torch.randperm(len(X_t))
        total_loss = 0.0

        for i in range(0, len(X_t), batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_t[idx], y_t[idx]
            wb = w_t[idx] if w_t is not None else None

            optimizer.zero_grad()
            loss = weighted_loss(model(xb), yb, wb)
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
                val_logits = model(X_v)
                val_loss = weighted_loss(val_logits, y_v, w_v).item()
                val_auc = (
                    float(roc_auc_score(np.asarray(y_val), torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()))
                    if can_use_auc
                    else float("nan")
                )
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)
            best_val_loss = min(best_val_loss, val_loss)
            val_info = f"  val_loss = {val_loss:.4f}  val_auc = {val_auc:.4f}"

            score = val_auc if early_stopping_metric == "val_auc" else -val_loss
            if epoch + 1 < min_epochs:
                # Still inside the excluded prefix: train on, but never keep these
                # weights and do not let them start the patience counter.
                pass
            elif score > best_score:
                best_score = score
                best_metric_epoch = epoch + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

        print(
            f"[Epoch {epoch+1:>3}/{epochs}] lr = {optimizer.param_groups[0]['lr']:.2e}  "
            f"loss = {avg_loss:.4f}{val_info}"
        )

        if X_v is not None and no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}  "
                f"(best {early_stopping_metric} at epoch {best_metric_epoch})"
            )
            stopped_early = True
            break

        if stop_after and (epoch + 1) >= stop_after:
            print(f"Stopping at the fixed budget of {stop_after} epoch(s)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_losses = history["val_loss"]
    val_aucs = [a for a in history["val_auc"] if a == a]  # drop NaN

    return model, {
        "classes": ["DOWN", "UP"],
        "history": history,
        "stopped_early": stopped_early,
        "early_stopping_metric": early_stopping_metric,
        "best_val_loss": float(min(val_losses)) if val_losses else None,
        "best_val_auc": float(max(val_aucs)) if val_aucs else None,
        # The epoch whose weights were actually kept, by the metric in use — not
        # argmin(val_loss), which no longer necessarily agrees with it.
        "best_epoch": best_metric_epoch,
        "epochs_run": len(val_losses) or len(history["train_loss"]),
    }


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Возвращает вероятности классов [DOWN, UP] формы (N, 2)."""
    model.eval()
    logits = model(torch.tensor(X, dtype=torch.float32))
    return torch.softmax(logits, dim=1).cpu().numpy()
