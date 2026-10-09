"""Train the legacy-compatible 2-8-2 EOG position network."""

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_id", help="identifier used in the training-data filenames")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "Pytorch" / "input_data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "Pytorch" / "output",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    feature_path = args.data_dir / f"input_{args.data_id}_filled.txt"
    target_path = args.data_dir / f"output_{args.data_id}_filled.txt"
    features = np.loadtxt(feature_path, delimiter=",", ndmin=2, dtype=np.float32)
    targets = np.loadtxt(target_path, delimiter=",", ndmin=2, dtype=np.float32)
    if features.shape != targets.shape or features.shape[1] != 2 or len(features) < 2:
        parser.error("input and output files must have matching N x 2 data with N >= 2")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        parser.error("training files contain non-finite values")

    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8),
        torch.nn.Sigmoid(),
        torch.nn.Linear(8, 2),
        torch.nn.Sigmoid(),
    ).to(device)
    x = torch.from_numpy(features).to(device)
    y = torch.from_numpy(targets).to(device)
    split = max(1, int(round(len(features) * 0.7)))
    x_train, y_train = x[:split], y[:split]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        prediction = model(x_train)
        loss = loss_function(prediction, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 1000 == 0 or epoch + 1 == args.epochs:
            print(f"epoch {epoch + 1}/{args.epochs}: loss={loss.item():.6f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = [layer for layer in model if isinstance(layer, torch.nn.Linear)]
    suffix = f"{args.data_id}.txt"
    with torch.no_grad():
        np.savetxt(
            args.output_dir / f"pred_{suffix}", model(x).cpu().numpy(), delimiter=","
        )
        np.savetxt(
            args.output_dir / f"w1_{suffix}", weights[0].weight.cpu().numpy(), delimiter=","
        )
        np.savetxt(
            args.output_dir / f"bias1_{suffix}",
            weights[0].bias.cpu().numpy(),
            delimiter=",",
        )
        np.savetxt(
            args.output_dir / f"w2_{suffix}", weights[1].weight.cpu().numpy(), delimiter=","
        )
        np.savetxt(
            args.output_dir / f"bias2_{suffix}",
            weights[1].bias.cpu().numpy(),
            delimiter=",",
        )
    print(f"Training complete. Weights saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
