"""
Unsupervised learning benchmark methods for anomaly detection.

Author: PoSeiDon Team
License: MIT
"""
from .pyod import *
from .pygod import *


def list_models():
    r""" List of models from `pyod.models`.

    Returns:
        list: List of models.
    """
    return list_pyod() + list_pygod()
