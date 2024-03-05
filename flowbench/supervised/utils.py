from lightning.pytorch.callbacks import EarlyStopping


def early_stopping_callback(minitor='val_loss', patience=5, mode='min'):
    return EarlyStopping(monitor=minitor, patience=patience, mode=mode)
