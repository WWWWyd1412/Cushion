"""
VME (变分模态提取) 信号提取
===========================
单模态变分提取，支持基线漂移去除和心跳主频提取。
"""

import numpy as np


def VME_Core(signal, fs, f_init=0.25, alpha=1000, tol=1e-6, max_iter=150):
    """
    变分模态提取 (VME) 核心算法。

    用于提取特定频率的单一变分模态分量。
    - 低频: f_init=0.25 Hz → 提取呼吸/基线漂移干扰 → 用于去除基线
    - 高频: f_init ≈ 1.2 Hz → 提取心跳主频模态

    引入镜像延拓消除边界瞬态畸变。
    """
    T = len(signal)
    if T < 10:
        return np.zeros_like(signal)

    # 镜像延拓
    pad_len = min(T // 2, 100)
    left_pad = signal[1:pad_len + 1][::-1]
    right_pad = signal[-pad_len - 1:-1][::-1]
    padded_signal = np.concatenate((left_pad, signal, right_pad))
    T_pad = len(padded_signal)

    # 构造解析信号的傅里叶变换
    f_fft = np.fft.fft(padded_signal)
    half_T = T_pad // 2
    f_fft_analytic = np.zeros_like(f_fft, dtype=complex)
    f_fft_analytic[0] = f_fft[0]
    f_fft_analytic[1:half_T] = 2.0 * f_fft[1:half_T]
    if T_pad % 2 == 0:
        f_fft_analytic[half_T] = f_fft[half_T]

    freqs = np.fft.fftfreq(T_pad, 1 / fs)

    u_fft = np.zeros(T_pad, dtype=complex)
    lambda_fft = np.zeros(T_pad, dtype=complex)
    omega_d = f_init
    tau = 0.1

    for it in range(max_iter):
        u_fft_old = u_fft.copy()

        diff = freqs[:half_T + 1] - omega_d
        diff2 = diff ** 2
        diff4 = diff2 ** 2

        num = (f_fft_analytic[:half_T + 1] +
               (alpha ** 2) * diff4 * u_fft_old[:half_T + 1] +
               lambda_fft[:half_T + 1] / 2.0)
        den = (1.0 + (alpha ** 2) * diff4) * (1.0 + 2.0 * alpha * diff2)

        u_fft[:half_T + 1] = num / (den + 1e-12)
        u_fft[half_T + 1:] = 0.0

        u_power = np.abs(u_fft[:half_T + 1]) ** 2
        sum_power = np.sum(u_power)
        if sum_power > 1e-12:
            omega_d = np.sum(freqs[:half_T + 1] * u_power) / sum_power

        error = ((f_fft_analytic[:half_T + 1] - u_fft[:half_T + 1]) /
                 (1.0 + (alpha ** 2) * diff4 + 1e-12))
        lambda_fft[:half_T + 1] = lambda_fft[:half_T + 1] + tau * error

        if it > 5:
            change = (np.linalg.norm(u_fft[:half_T + 1] - u_fft_old[:half_T + 1]) /
                      (np.linalg.norm(u_fft_old[:half_T + 1]) + 1e-12))
            if change < tol:
                break

    u_time_padded = np.real(np.fft.ifft(u_fft))
    u_time = u_time_padded[pad_len:pad_len + T]
    return u_time


def extract_vme(frames, fs):
    """
    VME 心跳节律提取 (直接定位心跳主频并提取模态)。
    """
    from cushion.algorithms.base import get_dual_roi_mean

    signal_1d = get_dual_roi_mean(frames, fs=fs,
                                   freq_band=(0.8, 2.2),
                                   wavelet_alpha=0.3,
                                   use_vme_baseline=True)
    if len(signal_1d) == 0:
        return np.zeros(100)

    signal_1d = signal_1d - np.mean(signal_1d)

    # 估计心跳主频
    n_len = len(signal_1d)
    f_dom = 1.2
    if n_len > 8:
        fft_vals = np.abs(np.fft.fft(signal_1d))[:n_len // 2]
        freqs_arr = np.fft.fftfreq(n_len, 1 / fs)[:n_len // 2]
        valid_mask = (freqs_arr >= 0.8) & (freqs_arr <= 2.2)
        if np.any(valid_mask):
            f_dom = freqs_arr[valid_mask][np.argmax(fft_vals[valid_mask])]

    return VME_Core(signal_1d, fs, f_init=f_dom, alpha=2000)


# 内部别名，供 base.py 中的 get_dual_roi_mean 使用
_vme_core = VME_Core
