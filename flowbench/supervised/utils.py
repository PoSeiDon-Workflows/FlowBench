""" Utility functions for supervised learning.

Author: PoSeiDon Team
License: MIT
"""
from lightning.pytorch.callbacks import EarlyStopping


def early_stopping_callback(minitor='val_loss', patience=5, mode='min'):
    r""" Early stopping callback.
    Args:
        minitor (str): The metric to monitor.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        mode (str): One of {'min', 'max'}.
            In 'min' mode, training will stop when the quantity monitored has stopped decreasing, e.g., validation loss;
            in 'max' mode it will stop when the quantity monitored has stopped increasing, e.g., validation accuracy.
    Return:
        EarlyStopping: Early stopping callback.
    """
    return EarlyStopping(monitor=minitor, patience=patience, mode=mode)
