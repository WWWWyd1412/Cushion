"""
EMD (经验模态分解) 信号提取
==========================
统一的 EMD 提取函数，通过参数化支持呼吸和心跳两种模式。
"""

import numpy as np
from PyEMD import EMD
from cushion.algorithms.base import (
    get_dual_roi_mean,
    get_spatial_sum,
    select_best_component,
    reconstruct_multicomponent_with_snr,
)


def extract_emd(frames, fs,
                freq_band=(0.1, 0.5),
                wavelet_alpha=0.5,
                use_vme_baseline=False,
                use_multicomponent=True,
                roi_mode='dual'):
    """
    EMD 信号提取。

    Parameters
    ----------
    frames : ndarray (N, 32, 32)
    fs : float
    freq_band : tuple
        目标频带 (low, high) Hz。
    wavelet_alpha : float
    use_vme_baseline : bool
        心跳模式: 启用 VME 基线漂移去除。
    use_multicomponent : bool
        True: 多分量 SNR 重构, False: 单分量最优选择。
    roi_mode : str
        'dual' (Breath/Heartbeat) 或 'spatial_sum' (RealTime)。

    Returns
    -------
    ndarray — 1D 提取信号。
    """
    if roi_mode == 'spatial_sum':
        signal_1d = get_spatial_sum(frames)
    else:
        signal_1d = get_dual_roi_mean(frames, fs=fs,
                                       freq_band=freq_band,
                                       wavelet_alpha=wavelet_alpha,
                                       use_vme_baseline=use_vme_baseline)
    if len(signal_1d) == 0:
        return np.zeros(100)

    emd = EMD()
    imfs = emd(signal_1d)

    if use_multicomponent:
        return reconstruct_multicomponent_with_snr(imfs, fs, freq_band=freq_band)
    return select_best_component(imfs, fs, freq_band=freq_band)
