"""
Multi-ROI ICA (独立成分分析) 信号融合
=====================================
FastICA 盲源分离 + 生理频段分量筛选 + 相位对齐。
"""

import numpy as np
from scipy.fft import fft, fftfreq
from sklearn.decomposition import FastICA
from cushion.core.signal_utils import calculate_snr
from cushion.algorithms.fusion.pca import fuse_signals_pca


def fuse_signals_ica(multi_channel_signals, fs=10.0, freq_band=(0.1, 0.5)):
    """
    FastICA 多通道盲源分离融合。

    对多 ROI 通道信号进行 ICA 解调，筛选出生理特征最典型的分量。

    Parameters
    ----------
    multi_channel_signals : ndarray, shape (M, T)
    fs : float
    freq_band : tuple (low, high)

    Returns
    -------
    ndarray, 1D — 融合后的单通道信号。
    """
    M, T = multi_channel_signals.shape
    if M == 1:
        return multi_channel_signals[0]

    X = multi_channel_signals.T
    n_comps = min(3, M)

    ica = FastICA(n_components=n_comps, random_state=42, max_iter=1000, tol=1e-3)
    try:
        sources = ica.fit_transform(X)
    except Exception as e:
        print(f"[ICA Warning] FastICA 迭代异常: {e}. 自动回退至 PCA 融合。")
        return fuse_signals_pca(multi_channel_signals)

    best_source = None
    max_snr = -999.0

    for k in range(n_comps):
        comp = sources[:, k]
        n = len(comp)
        freqs = fftfreq(n, 1 / fs)[:n // 2]
        fft_vals = np.abs(fft(comp))[:n // 2]
        dom_freq = freqs[np.argmax(fft_vals)]

        if freq_band[0] <= dom_freq <= freq_band[1]:
            snr = calculate_snr(comp, fs, band=freq_band)
            if snr > max_snr:
                max_snr = snr
                best_source = comp

    if best_source is not None:
        # 相位对齐
        channel_mean = np.mean(multi_channel_signals, axis=0)
        if np.dot(best_source, channel_mean) < 0:
            best_source = -best_source
        return best_source
    else:
        return sources[:, 0]


def extract_multi_roi_ica(frames, fs):
    """
    Multi-ROI ICA 呼吸信号提取入口。

    1. 4 象限 ROI 提取 + 去噪 + 带通滤波
    2. FastICA 盲源分离融合
    """
    from cushion.algorithms.base import get_multi_roi_signals

    multi_signals = get_multi_roi_signals(frames, num_rois=4, window_size=5)
    if len(multi_signals) == 0:
        return np.zeros(len(frames))

    return fuse_signals_ica(multi_signals, fs)
