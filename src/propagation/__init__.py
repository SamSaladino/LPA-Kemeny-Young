"""Label propagation and candidate aggregation."""

from .kemeny import (
    KemenyYoung,
    lpa,
    run_label_propagation,
    set_labels,
)

__all__ = [
    "KemenyYoung",
    "lpa",
    "run_label_propagation",
    "set_labels",
]