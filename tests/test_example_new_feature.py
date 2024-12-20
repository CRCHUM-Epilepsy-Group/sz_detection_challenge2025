import numpy as np

from szdetect import example_new_feature as nf


def test_new_feature():
    # Connectivity matrix of shape (n_epochs, n_channels, n_channels)
    con_mat = np.ones((100, 19, 19))

    new_f = nf.new_feature(con_mat)
    assert new_f.shape[0] == 100
