"""The same network as model.py, built in Keras.

Two hidden layers with ReLU, ending in one value with no activation,
because the target is a count of rentals, not a probability.
"""

import keras
from torch import nn


def build_model(n_features: int, hidden: int = 32) -> keras.Model:
    """Build the Keras version of BikeNet with the given hidden size."""
    model = keras.Sequential(
        [
            keras.Input(shape=(n_features,)),
            keras.layers.Dense(hidden, activation="relu"),
            keras.layers.Dense(hidden, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    return model


def copy_weights(torch_model: nn.Module, keras_model: keras.Model) -> None:
    """Copy weights from a PyTorch BikeNet into the Keras model.

    A PyTorch weight has shape (outputs, inputs) and a Keras Dense
    kernel has shape (inputs, outputs), so each weight is transposed.
    The biases copy straight across.
    """
    state = torch_model.state_dict()
    dense = [
        layer for layer in keras_model.layers if isinstance(layer, keras.layers.Dense)
    ]
    names = ["layers.0", "layers.2", "layers.4"]
    for i, name in enumerate(names):
        weight = state[name + ".weight"].detach().numpy().T
        bias = state[name + ".bias"].detach().numpy()
        dense[i].set_weights([weight, bias])
