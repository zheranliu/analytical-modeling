"""
data_generator.py

Utilities for generating synthetic ("Hello World") datasets from known
analytical models.

This module is used throughout the CMSE 802 analytical modeling unit.
It supports activities involving:

- model identification
- parameter estimation
- optimization
- symbolic regression
- testing and validation

The key idea is that we can generate observations from a known model,
hide the generating equation, and then attempt to recover the model
from the resulting data.

Examples
--------
Generate a random dataset:

>>> data, truth = generate_dataset(seed=42)

Generate a specific model:

>>> data, truth = generate_dataset(
...     model_name="exponential",
...     seed=42
... )

Inspect the generated data:

>>> data.head()

View the hidden model information:

>>> truth

Notes
-----
The returned ``truth`` dictionary contains the generating information.
In classroom activities instructors may choose not to provide this
information to students.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "available_models",
    "generate_dataset",
    "evaluate_truth",
    "generate_truth_dataset",
]


def exponential(
    x: np.ndarray,
    A: float,
    k: float,
) -> np.ndarray:
    """
    Evaluate an exponential decay model.

    Parameters
    ----------
    x
        Independent variable values.

    A
        Initial value or scale factor.

    k
        Decay rate.

    Returns
    -------
    numpy.ndarray
        Model predictions.

    Notes
    -----
    The model has the form

    .. math::

        y = A e^{-kx}

    This model appears in many contexts including:

    - radioactive decay
    - drug metabolism
    - cooling processes
    - charging and discharging systems
    """
    return A * np.exp(-k * x)


def linear(
    x: np.ndarray,
    A: float,
    B: float,
) -> np.ndarray:
    """
    Evaluate a linear model.

    Parameters
    ----------
    x
        Independent variable values.

    A
        Intercept.

    B
        Slope.

    Returns
    -------
    numpy.ndarray
        Model predictions.

    Notes
    -----
    The model has the form

    .. math::

        y = A + Bx
    """
    return A + B * x


def quadratic(
    x: np.ndarray,
    A: float,
    B: float,
    C: float,
) -> np.ndarray:
    """
    Evaluate a quadratic model.

    Parameters
    ----------
    x
        Independent variable values.

    A
        Quadratic coefficient.

    B
        Linear coefficient.

    C
        Constant term.

    Returns
    -------
    numpy.ndarray
        Model predictions.

    Notes
    -----
    The model has the form

    .. math::

        y = Ax^2 + Bx + C
    """
    return A * x**2 + B * x + C


def logarithmic(
    x: np.ndarray,
    A: float,
    B: float,
) -> np.ndarray:
    """
    Evaluate a logarithmic model.

    Parameters
    ----------
    x
        Independent variable values.

    A
        Scale factor.

    B
        Vertical offset.

    Returns
    -------
    numpy.ndarray
        Model predictions.

    Notes
    -----
    The model has the form

    .. math::

        y = A\\log(x) + B
    """
    return A * np.log(x) + B


MODELS = {
    "exponential": {
        "function": exponential,
        "parameter_generator": lambda rng: {
            "A": rng.uniform(10, 100),
            "k": rng.uniform(0.05, 0.50),
        },
    },
    "linear": {
        "function": linear,
        "parameter_generator": lambda rng: {
            "A": rng.uniform(-10, 10),
            "B": rng.uniform(1, 10),
        },
    },
    "quadratic": {
        "function": quadratic,
        "parameter_generator": lambda rng: {
            "A": rng.uniform(0.1, 2.0),
            "B": rng.uniform(-5.0, 5.0),
            "C": rng.uniform(-20.0, 20.0),
        },
    },
    "logarithmic": {
        "function": logarithmic,
        "parameter_generator": lambda rng: {
            "A": rng.uniform(1.0, 20.0),
            "B": rng.uniform(-10.0, 10.0),
        },
    },
}


def available_models() -> list[str]:
    """Return the names of available model families."""
    return sorted(MODELS.keys())


def generate_dataset(
    model_name: str | None = None,
    seed: int = 42,
    n_points: int = 50,
    noise_fraction: float = 0.10,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Generate a synthetic dataset.

    Parameters
    ----------
    model_name
        Name of the model family to use. If ``None``,
        a model is selected randomly.

    seed
        Random seed used for repeatable dataset generation.

    n_points
        Number of observations to generate.

    noise_fraction
        Standard deviation of the noise as a fraction of the
        model's natural variation.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing columns:

        - ``x``
        - ``y``

    dict
        Dictionary containing the hidden generating information.

    Examples
    --------
    Generate an exponential dataset:

    >>> data, truth = generate_dataset(
    ...     model_name="exponential"
    ... )

    Generate a random dataset:

    >>> data, truth = generate_dataset()

    Notes
    -----
    The returned metadata dictionary contains information
    that would normally be hidden from students during
    model discovery exercises.
    """
    rng = np.random.default_rng(seed)

    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    if noise_fraction < 0:
        raise ValueError("noise_fraction must be non-negative")

    if model_name is None:
        model_name = str(rng.choice(list(MODELS.keys())))

    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. " f"Available models: {available_models()}"
        )

    model_info = MODELS[model_name]

    parameters = model_info["parameter_generator"](rng)

    x = np.linspace(1.0, 20.0, n_points)

    true_y = model_info["function"](x, **parameters)

    noise_scale = noise_fraction * np.std(true_y)

    observed_y = true_y + rng.normal(
        loc=0.0,
        scale=noise_scale,
        size=n_points,
    )

    data = pd.DataFrame(
        {
            "x": x,
            "y": observed_y,
        }
    )

    truth = {
        "model_name": model_name,
        "parameters": parameters,
        "seed": seed,
        "noise_fraction": noise_fraction,
        "n_points": n_points,
    }

    return data, truth


def evaluate_truth(
    x: np.ndarray,
    truth: dict,
) -> np.ndarray:
    """
    Evaluate the hidden model stored in a truth dictionary.

    Parameters
    ----------
    x
        Independent variable values.

    truth
        Dictionary returned by ``generate_dataset``.

    Returns
    -------
    numpy.ndarray
        Noise-free model predictions.

    Examples
    --------
    >>> data, truth = generate_dataset(seed=42)
    >>> y = evaluate_truth(data["x"], truth)
    """
    model_name = truth["model_name"]

    model_function = MODELS[model_name]["function"]

    return model_function(
        np.asarray(x),
        **truth["parameters"],
    )


def generate_truth_dataset(
    truth: dict,
    n_points: int | None = None,
) -> pd.DataFrame:
    """
    Generate a noise-free dataset from a truth dictionary.

    Parameters
    ----------
    truth
        Dictionary returned by ``generate_dataset``.

    n_points
        Number of points to generate. If ``None``, the same
        number of points used during generation is used.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing:

        - x
        - y

        representing the underlying noise-free model.

    Examples
    --------
    >>> data, truth = generate_dataset()
    >>> truth_data = generate_truth_dataset(truth)
    """

    if n_points is None:
        n_points = truth.get("n_points", 50)

    x = np.linspace(1.0, 20.0, n_points)

    y = evaluate_truth(x, truth)

    return pd.DataFrame(
        {
            "x": x,
            "y": y,
        }
    )
