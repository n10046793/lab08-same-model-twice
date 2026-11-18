"""Train the Keras version of the network and write every result.

Run with: python train_keras.py

This uses the same split and the same prepared arrays as train.py,
the same loss, optimizer, learning rate, batch size and seed, and
writes learning_curves_keras.png and history_keras.csv beside the
PyTorch reports. The seed is fixed, so running it twice on the same
computer gives the same numbers.
"""

import csv
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from keras_model import build_model
from prepare import BATCH_SIZE, SEED, make_splits

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
EPOCHS = 80
LEARNING_RATE = 3e-3
HIDDEN = 32


def plot_curves(history, path, best_epoch):
    """Plot train and val loss per epoch and mark the best epoch."""
    epochs = list(range(1, len(history["loss"]) + 1))
    plt.figure()
    plt.plot(epochs, history["loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.axvline(best_epoch, color="grey", linestyle="--", label="best epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss (mean squared error)")
    plt.title("Learning curves (Keras)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    keras.utils.set_random_seed(SEED)
    REPORTS.mkdir(exist_ok=True)

    splits = make_splits(SEED)
    print(f"train {len(splits.train)} val {len(splits.val)} test {len(splits.test)}")
    print(f"features per row: {splits.n_features}")

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

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EPOCHS,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            str(REPORTS / "best_keras_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
    )

    # The history records one row per epoch. The best epoch is the one
    # with the lowest validation loss, the same rule as train.py.
    log = history.history
    best_epoch = int(np.argmin(log["val_loss"])) + 1
    rows = [
        {
            "epoch": epoch,
            "train_loss": log["loss"][epoch - 1],
            "val_loss": log["val_loss"][epoch - 1],
            "val_mae": log["val_mae"][epoch - 1],
        }
        for epoch in range(1, len(log["loss"]) + 1)
    ]
    plot_curves(log, REPORTS / "learning_curves_keras.png", best_epoch)
    with open(REPORTS / "history_keras.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_loss", "val_mae"]
        )
        writer.writeheader()
        writer.writerows(rows)

    # EarlyStopping put back the best weights, so this scores the best
    # epoch's model, the same as train.py does.
    _, val_mae = model.evaluate(x_val, y_val, verbose=0)
    _, test_mae = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nBest epoch: {best_epoch}")
    print(f"Keras val MAE: {val_mae:.2f} test MAE: {test_mae:.2f}")
    print("\nDone. Reports are in reports/.")


if __name__ == "__main__":
    main()
