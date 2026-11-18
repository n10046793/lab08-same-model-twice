"""Tests for the Keras version of the network.

The Keras network must match the PyTorch network: same shape, same
parameter count, same predictions once the weights are copied, and
the same rows. The weight-copy tolerance is 1e-4, far above the
3.7e-08 difference measured on the validation rows.
"""

import keras
import numpy as np
import pytest
import torch

from keras_model import build_model, copy_weights
from model import BikeNet
from prepare import SEED, make_loaders, make_splits

TOLERANCE = 1e-4


def test_keras_net_returns_one_value_per_row():
    model = build_model(8)
    rows = np.zeros((5, 8), dtype=np.float32)
    predictions = model.predict(rows, verbose=0)
    assert predictions.shape == (5, 1)
    assert len(predictions.flatten()) == 5


@pytest.mark.parametrize("hidden", [8, 16, 32, 64])
def test_parameter_count_matches_pytorch(hidden):
    torch_model = BikeNet(8, hidden=hidden)
    torch_count = sum(p.numel() for p in torch_model.parameters())
    keras_model = build_model(8, hidden=hidden)
    assert keras_model.count_params() == torch_count


def test_weight_copy_gives_same_predictions():
    # Validation rows have distinct values, so using a weight the
    # wrong way round changes the answer. This matters most for
    # the middle layer, whose weight is square and would fit either
    # way round without raising an error.
    splits = make_splits(SEED)
    rows = splits.val.x.numpy()[:24]
    torch.manual_seed(11)
    torch_model = BikeNet(8)
    keras_model = build_model(8)
    copy_weights(torch_model, keras_model)
    torch_model.eval()
    torch_predictions = torch_model(torch.from_numpy(rows)).detach().numpy()
    keras_predictions = keras_model.predict(rows, verbose=0).flatten()
    np.testing.assert_allclose(keras_predictions, torch_predictions, atol=TOLERANCE)


def test_both_frameworks_use_the_same_rows():
    # Keras trains on the prepared arrays directly. PyTorch reads the
    # same prepared arrays through a loader. The rows must match.
    splits = make_splits(SEED)
    _, val_loader, _ = make_loaders(splits, 64, SEED)
    loader_x = np.concatenate([x.numpy() for x, _ in val_loader])
    loader_y = np.concatenate([y.numpy() for _, y in val_loader])
    np.testing.assert_array_equal(loader_x, splits.val.x.numpy())
    np.testing.assert_array_equal(loader_y, splits.val.y.numpy())


def test_keras_net_can_memorize_one_small_batch():
    rng = np.random.default_rng(3)
    rows = rng.normal(size=(8, 8)).astype(np.float32)
    targets = (rows[:, 0] * 2 + 1).astype(np.float32)
    keras.utils.set_random_seed(SEED)
    model = build_model(8)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="mse",
    )
    model.fit(rows, targets, epochs=200, verbose=0)
    predictions = model.predict(rows, verbose=0).flatten()
    assert np.mean(np.abs(predictions - targets)) < 0.1


def test_saved_and_reloaded_model_gives_same_predictions(tmp_path):
    rows = np.random.default_rng(5).normal(size=(6, 8)).astype(np.float32)
    keras.utils.set_random_seed(SEED)
    model = build_model(8)
    before = model.predict(rows, verbose=0)
    path = tmp_path / "model.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    after = loaded.predict(rows, verbose=0)
    np.testing.assert_allclose(after, before, atol=1e-5)


def test_best_weights_are_restored():
    splits = make_splits(SEED)
    keras.utils.set_random_seed(SEED)
    model = build_model(splits.n_features)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-3),
        loss="mse",
        metrics=["mae"],
    )
    early = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )
    history = model.fit(
        splits.train.x.numpy(),
        splits.train.y.numpy(),
        batch_size=64,
        epochs=10,
        validation_data=(splits.val.x.numpy(), splits.val.y.numpy()),
        callbacks=[early],
        verbose=0,
    )
    # The restored model must score the best epoch's validation loss.
    val_loss, _ = model.evaluate(splits.val.x.numpy(), splits.val.y.numpy(), verbose=0)
    best_loss = min(history.history["val_loss"])
    assert abs(val_loss - best_loss) < 1e-3


def test_same_seed_repeats_a_short_run():
    splits = make_splits(SEED)
    rows = splits.train.x.numpy()[:256]
    targets = splits.train.y.numpy()[:256]
    check = splits.val.x.numpy()[:16]

    def short_run():
        keras.utils.set_random_seed(SEED)
        model = build_model(splits.n_features)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-3),
            loss="mse",
        )
        model.fit(rows, targets, batch_size=64, epochs=3, verbose=0)
        return model.predict(check, verbose=0)

    first = short_run()
    second = short_run()
    np.testing.assert_allclose(second, first, atol=1e-5)
