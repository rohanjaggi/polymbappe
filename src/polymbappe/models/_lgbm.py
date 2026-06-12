"""Small helpers for LightGBM's scikit-learn estimators."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def silence_feature_name_warning() -> Iterator[None]:
    """Suppress LightGBM's spurious sklearn feature-name-mismatch warning.

    LightGBM synthesizes placeholder feature names (``Column_0`` …) and records them in
    ``feature_names_in_`` even when fit on a bare numpy array, so it is indistinguishable
    from an estimator fit on a named DataFrame. sklearn's input validation then emits
    ``"X does not have valid feature names, but <Estimator> was fitted with feature names"``
    on every numpy-array ``predict`` — a false positive here, because our fit and predict
    paths both build the matrix from the *same* ordered feature-column list, so the columns
    can never silently misalign. Silence only that one message; let everything else through.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        yield
