"""
PCA (主成分分析) 信号融合
=========================
PCA 降维融合，作为 ICA 的保底方案。
"""

import numpy as np
from sklearn.decomposition import PCA


def fuse_signals_pca(multi_channel_signals):
    """
    PCA 单成分融合。

    Parameters
    ----------
    multi_channel_signals : ndarray, shape (M, T)

    Returns
    -------
    ndarray, 1D
    """
    X = multi_channel_signals.T
    pca = PCA(n_components=1, random_state=42)
    source = pca.fit_transform(X).flatten()

    channel_mean = np.mean(multi_channel_signals, axis=0)
    if np.dot(source, channel_mean) < 0:
        source = -source
    return source
