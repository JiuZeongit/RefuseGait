"""Losses required by DAVIS346-Gait ReFuseGait training."""

from .base import BaseLoss
from .ce import CrossEntropyLoss
from .triplet import TripletLoss

__all__ = [
    "BaseLoss",
    "CrossEntropyLoss",
    "TripletLoss",
]
