"""Train the main configuration in both frameworks on three seeds.

Run with: python compare.py

Each seed makes one split, and both frameworks train on that same
split with the same prepared arrays, the same loss, optimizer,
learning rate, batch size and number of epochs. Each final model is
scored on the test rows exactly once. The seconds column is training
time only. The table goes to reports/comparison.csv.
"""

import csv
import time
from pathlib import Path

import keras
import numpy as np
import torch
from torch import nn

from keras_model import build_model
from model import BikeNet
from prepare import BATCH_SIZE, make_loaders, make_splits
from train import run_epoch, train_model

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
EPOCHS = 80
LEARNING_RATE = 3e-3
HIDDEN = 32
SEEDS = [42, 123, 2026]


def train_torch(splits, seed):
    """Train the PyTorch network. Returns best epoch, val MAE, test MAE, seconds."""
    torch.manual_seed(seed)
    train_loader, val_loader, test_loader = make_loaders(splits, BATCH_SIZE, seed)
    model = BikeNet(splits.n_features, hidden=HIDDEN)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start = time.perf_counter()
    history, best_epoch = train_model(
        model, train_loader, val_loader, loss_fn, optimizer, EPOCHS
    )
    seconds = time.perf_counter() - start
    best_row = min(history, key=lambda row: row["val_loss"])
    _, test_mae = run_epoch(model, test_loader, loss_fn)
    return best_epoch, best_row["val_mae"], test_mae, seconds


def train_keras(splits, seed):
    """Train the Keras network. Returns best epoch, val MAE, test MAE, seconds."""
    keras.utils.set_random_seed(seed)
    x_train = splits.train.x.numpy()
    y_train = splits.train.y.numpy()
    x_val = splits.val.x.numpy()
    y_val = splits.val.y.numpy()
    x_test = splits.test.x.numpy()
    y_test = splits.test.y.numpy()

    model = build_model(splits.n_features, hidden=HIDDEN)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    early = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EPOCHS,
        restore_best_weights=True,
    )
    start = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=[early],
        verbose=0,
    )
    seconds = time.perf_counter() - start
    log = history.history
    best_epoch = int(np.argmin(log["val_loss"])) + 1
    val_mae = log["val_mae"][best_epoch - 1]
    _, test_mae = model.evaluate(x_test, y_test, verbose=0)
    return best_epoch, val_mae, test_mae, seconds


def main():
    REPORTS.mkdir(exist_ok=True)
    rows = []
    for seed in SEEDS:
        splits = make_splits(seed)
        print(f"\nSeed {seed}: train {len(splits.train)} "
              f"val {len(splits.val)} test {len(splits.test)}")
        best_epoch, val_mae, test_mae, seconds = train_torch(splits, seed)
        rows.append(
            {
                "framework": "pytorch",
                "seed": seed,
                "best_epoch": best_epoch,
                "val_mae": val_mae,
                "test_mae": test_mae,
                "seconds": seconds,
            }
        )
        print(
            f"pytorch seed {seed}: best epoch {best_epoch} "
            f"val MAE {val_mae:.2f} test MAE {test_mae:.2f} "
            f"seconds {seconds:.1f}"
        )
        best_epoch, val_mae, test_mae, seconds = train_keras(splits, seed)
        rows.append(
            {
                "framework": "keras",
                "seed": seed,
                "best_epoch": best_epoch,
                "val_mae": val_mae,
                "test_mae": test_mae,
                "seconds": seconds,
            }
        )
        print(
            f"keras seed {seed}: best epoch {best_epoch} "
            f"val MAE {val_mae:.2f} test MAE {test_mae:.2f} "
            f"seconds {seconds:.1f}"
        )

    with open(REPORTS / "comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "framework",
                "seed",
                "best_epoch",
                "val_mae",
                "test_mae",
                "seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print("\nDone. The table is in reports/comparison.csv.")


if __name__ == "__main__":
    main()
