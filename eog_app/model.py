"""Training and inference for the two-input EOG position network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def train_model(
    features: NDArray[np.generic],
    positions: NDArray[np.generic],
    output_path: Path,
    *,
    epochs: int = 3000,
    learning_rate: float = 1e-3,
    seed: int = 7,
) -> float:
    """Train the original 2-8-2 sigmoid network on CPU and save its weights."""
    import torch

    x_values = np.asarray(features, dtype=np.float32)
    y_values = np.asarray(positions, dtype=np.float32)
    if x_values.ndim != 2 or x_values.shape[1] != 2:
        raise ValueError("training features must have shape (N, 2)")
    if y_values.shape != x_values.shape or x_values.shape[0] < 2:
        raise ValueError("training targets must match features and contain 2+ rows")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("training data must contain only finite values")
    if epochs < 1:
        raise ValueError("epochs must be positive")

    torch.manual_seed(seed)
    x = torch.from_numpy(x_values)
    y = torch.from_numpy(y_values)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8),
        torch.nn.Sigmoid(),
        torch.nn.Linear(8, 2),
        torch.nn.Sigmoid(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss(reduction="mean")

    for _ in range(epochs):
        prediction = model(x)
        loss = loss_function(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    layers = [layer for layer in model if isinstance(layer, torch.nn.Linear)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        w1=layers[0].weight.detach().numpy(),
        bias1=layers[0].bias.detach().numpy(),
        w2=layers[1].weight.detach().numpy(),
        bias2=layers[1].bias.detach().numpy(),
    )
    return float(loss.detach().item())


def predict(features: NDArray[np.generic], model_path: Path) -> NDArray[np.float64]:
    """Apply the saved sigmoid network to one or more feature rows."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("prediction input must have shape (N, 2)")
    if not np.isfinite(values).all():
        raise ValueError("prediction input must contain only finite values")

    with np.load(model_path, allow_pickle=False) as model:
        hidden = _sigmoid(values @ model["w1"].T + model["bias1"])
        output = _sigmoid(hidden @ model["w2"].T + model["bias2"])
    return np.asarray(output, dtype=np.float64)


def _sigmoid(values: NDArray[np.generic]) -> NDArray[np.float64]:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))
