"""
MVMD (多元变分模态分解) 信号提取
===============================
多通道联合 VMD 分解，结合 FastICA 进行时空盲源分离。
"""

import numpy as np
from scipy.fft import fft, ifft
from cushion.algorithms.base import (
    get_multi_roi_signals,
    reconstruct_multicomponent_with_snr,
)
from cushion.algorithms.fusion.ica import fuse_signals_ica


def mvmd(X, alpha=2000, tau=0, K=4, DC=0, init=1, tol=1e-7, max_iter=200):
    """
    多元变分模态分解 (Multivariate VMD)。

    Parameters
    ----------
    X : ndarray, shape (M, T)
        多通道输入信号。
    alpha : float
        带宽约束参数。
    K : int
        模态分解个数。
    tol : float
        收敛阈值。
    max_iter : int
        最大迭代次数。

    Returns
    -------
    u_time : ndarray, shape (K, M, T)
    omega : ndarray, shape (N_iter, K)
    """
    M, T = X.shape

    # 镜像延拓
    half_T = T // 2
    X_padded = np.zeros((M, 2 * T))
    for m in range(M):
        X_padded[m, :half_T] = X[m, :half_T][::-1]
        X_padded[m, half_T:half_T + T] = X[m, :]
        X_padded[m, half_T + T:] = X[m, T - half_T:][::-1]

    T_padded = X_padded.shape[1]
    freqs = np.arange(T_padded) / T_padded

    X_fft = fft(X_padded, axis=1)
    half_len = T_padded // 2
    X_fft_half = X_fft[:, :half_len]
    freqs_half = freqs[:half_len]

    # 初始化中心频率
    omega = np.zeros((max_iter, K))
    if init == 1:
        for k in range(K):
            omega[0, k] = 0.5 * k / K
    elif init == 2:
        omega[0, :] = np.logspace(np.log10(0.01), np.log10(0.5), K)
    else:
        omega[0, :] = np.sort(np.random.rand(K) * 0.5)

    if DC == 1:
        omega[0, 0] = 0.0

    u_fft = np.zeros((K, M, half_len), dtype=complex)
    lambda_fft = np.zeros((M, half_len), dtype=complex)

    it = 0
    converged = False

    while it < max_iter - 1 and not converged:
        u_fft_old = u_fft.copy()

        for k in range(K):
            sum_other = np.sum(u_fft, axis=0) - u_fft[k, :, :]
            denom = 1.0 + 2.0 * alpha * (freqs_half - omega[it, k]) ** 2

            for m in range(M):
                numerator = X_fft_half[m, :] - sum_other[m, :] - lambda_fft[m, :] / 2.0
                u_fft[k, m, :] = numerator / denom

            if not (DC == 1 and k == 0):
                power_spectrum = np.sum(np.abs(u_fft[k, :, :]) ** 2, axis=0)
                denom_freq = np.sum(power_spectrum)
                if denom_freq > 1e-12:
                    omega[it + 1, k] = np.sum(freqs_half * power_spectrum) / denom_freq
                else:
                    omega[it + 1, k] = omega[it, k]
            else:
                omega[it + 1, k] = 0.0

        lambda_fft += tau * (X_fft_half - np.sum(u_fft, axis=0))

        diff_sum = 0.0
        norm_sum = 0.0
        for k in range(K):
            diff_sum += np.sum(np.abs(u_fft[k] - u_fft_old[k]) ** 2)
            norm_sum += np.sum(np.abs(u_fft_old[k]) ** 2)
        if diff_sum / (norm_sum + 1e-12) < tol:
            converged = True

        it += 1

    u_fft_full = np.zeros((K, M, T_padded), dtype=complex)
    u_fft_full[:, :, :half_len] = u_fft

    u_time = np.zeros((K, M, T))
    for k in range(K):
        for m in range(M):
            analytic = ifft(u_fft_full[k, m, :])
            u_time[k, m, :] = 2.0 * np.real(analytic[half_T:half_T + T])

    return u_time, omega[:it, :]


def extract_mvmd(frames, fs, K=4, alpha=2000):
    """
    MVMD 多通道呼吸信号提取入口。

    1. 提取 4 个核心象限 ROI 信号
    2. MVMD 时空联合模态分解
    3. 每个模态进行 FastICA 融合
    4. SNR 多分量重构
    """
    multi_signals = get_multi_roi_signals(frames, num_rois=4, window_size=5)
    if len(multi_signals) == 0:
        return np.zeros(len(frames))

    u_time, _ = mvmd(multi_signals, alpha=alpha, tau=0, K=K, DC=0, init=1, tol=1e-7)

    fused_components = []
    for k in range(K):
        fused_k = fuse_signals_ica(u_time[k], fs)
        fused_components.append(fused_k)

    return reconstruct_multicomponent_with_snr(np.array(fused_components), fs)
