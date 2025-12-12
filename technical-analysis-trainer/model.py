import torch
import torch.nn as nn
import numpy as np

class TechnicalRRN(nn.Module):
    def __init__(self, hidden_size=50): # type: ignore
        super().__init__() # type: ignore
        self.lstm = nn.LSTM(input_size=5, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

def train_model(X: np.ndarray, y: np.ndarray, cfg: dict) -> tuple[nn.Module, dict]:
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)  # Add feature dimension
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Model parameters
    hidden_size = cfg.get("hidden_size", 50)
    learning_rate = cfg.get("learning_rate", 0.001)
    epochs = cfg.get("epochs", 100)

    # Initialize model, loss function, and optimizer
    model = TechnicalRRN(hidden_size=hidden_size)
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

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    train_metrics = {"final_loss": loss.item()} # type: ignore

    return model, train_metrics