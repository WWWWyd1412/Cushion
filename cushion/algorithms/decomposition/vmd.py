"""
VMD (变分模态分解) 信号提取
==========================
统一的 VMD 提取函数，通过参数化支持呼吸和心跳两种模式。
"""

import numpy as np
from vmdpy import VMD
from cushion.algorithms.base import (
    get_dual_roi_mean,
    get_spatial_sum,
    select_best_component,
    reconstruct_multicomponent_with_snr,
)


def extract_vmd(frames, fs,
                K=5, alpha=2000,
                freq_band=(0.1, 0.5),
                wavelet_alpha=0.5,
                use_vme_baseline=False,
                use_multicomponent=True,
                roi_mode='dual'):
    """
    VMD 信号提取。

    Parameters
    ----------
    frames : ndarray (N, 32, 32)
    fs : float
    K : int
        VMD 分解模态数。
    alpha : float
        VMD 带宽约束参数。
    freq_band : tuple
        目标频带 (low, high) Hz。
    wavelet_alpha : float
    use_vme_baseline : bool
        心跳模式: 启用 VME 基线漂移去除。
    use_multicomponent : bool
    roi_mode : str
        'dual' 或 'spatial_sum'。

    Returns
    -------
    ndarray — 1D 提取信号。
    """
    if roi_mode == 'spatial_sum':
        signal_1d = get_spatial_sum(frames)
        if len(signal_1d) < 100:
            return signal_1d
    else:
        signal_1d = get_dual_roi_mean(frames, fs=fs,
                                       freq_band=freq_band,
                                       wavelet_alpha=wavelet_alpha,
                                       use_vme_baseline=use_vme_baseline)
        if len(signal_1d) == 0:
            return np.zeros(100)

    u, _, _ = VMD(signal_1d, alpha=alpha, tau=0, K=K, DC=0, init=1, tol=1e-7)

    if use_multicomponent:
        return reconstruct_multicomponent_with_snr(u, fs, freq_band=freq_band)
    return select_best_component(u, fs, freq_band=freq_band)
