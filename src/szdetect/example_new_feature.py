import numpy as np


def new_feature(con_mat: np.ndarray) -> np.ndarray:
    """Average over the connectivity axes and return the features with shape (n_epochs, )"""
    return np.mean(con_mat, axis=(-2, -1))
