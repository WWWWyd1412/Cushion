"""
ACMD (自适应啁啾模式分解) 信号提取
=================================
针对心脉 (BCG) 信号的线性调频模态分解。
"""

import numpy as np
from cushion.algorithms.base import (
    get_dual_roi_mean,
    select_best_component,
)


def ACMD_Core(signal, fs, max_components=6, tol=1e-4):
    """
    ACMD 核心算法 — 自适应线性调频模态分解。

    Parameters
    ----------
    signal : ndarray, 1D
    fs : float
    max_components : int
    tol : float

    Returns
    -------
    components : ndarray, shape (K, N)
    residual : ndarray, 1D
    """
    components = []
    residual = signal.copy()
    orig_energy = np.sum(signal ** 2)

    for i in range(max_components):
        n = len(residual)
        fft_vals = np.abs(np.fft.fft(residual))[:n // 2]
        freqs = np.fft.fftfreq(n, 1 / fs)[:n // 2]

        valid_idx = (freqs >= 0.75) & (freqs <= 2.5)
        if np.any(valid_idx):
            init_freq = freqs[valid_idx][np.argmax(fft_vals[valid_idx])]
        else:
            init_freq = 1.2  # 72 BPM

        t = np.arange(n) / fs
        c = np.cos(2 * np.pi * init_freq * t)
        s = np.sin(2 * np.pi * init_freq * t)

        comp_i = (c * (np.dot(residual, c) / (np.dot(c, c) + 1e-6)) +
                   s * (np.dot(residual, s) / (np.dot(s, s) + 1e-6)))

        residual -= comp_i
        components.append(comp_i)

        if np.sum(residual ** 2) / (orig_energy + 1e-12) < tol:
            break

    return np.array(components), residual


def extract_acmd(frames, fs):
    """
    ACMD 心跳信号提取 (0.8-2.2 Hz)。
    """
    signal_1d = get_dual_roi_mean(frames, fs=fs,
                                   freq_band=(0.8, 2.2),
                                   wavelet_alpha=0.3,
                                   use_vme_baseline=True)
    if len(signal_1d) == 0:
        return np.zeros(100)

    signal_1d = signal_1d - np.mean(signal_1d)
    components, _ = ACMD_Core(signal_1d, fs, max_components=6, tol=0.0001)
    return select_best_component(components, fs, freq_band=(0.8, 2.2))
