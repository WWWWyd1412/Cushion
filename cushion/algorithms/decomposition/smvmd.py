"""
SMVMD (逐次多元变分模态分解) 信号提取
====================================
逐次提取多元信号中的模态分量，用于呼吸信号提取。
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
from cushion.core.signal_utils import calculate_snr


def SMVMD_Core(X_matrix, fs, alpha_min=1.0, alpha_max=2000.0, gamma=1.414,
               epsilon1=1e-7, epsilon2=1e-5, max_K=3, max_iter=150,
               progress_callback=None):
    """
    SMVMD 核心迭代算法 (高效向量化版)。

    Parameters
    ----------
    X_matrix : ndarray, shape (C, T)
        多通道输入信号矩阵。
    fs : float
        采样率。
    alpha_min, alpha_max : float
        带宽约束参数范围。
    gamma : float
        带宽递增因子。
    epsilon1, epsilon2 : float
        内/外层收敛阈值。
    max_K : int
        最大提取模态数。
    max_iter : int
        每个模态的最大迭代次数。
    progress_callback : callable or None
        callback(current_step, total_steps, mode_idx, iter_idx) -> bool
        返回 False 则中止计算。

    Returns
    -------
    u_list : list of ndarray — 时域模态分量列表
    phi_list : list of ndarray — 空间混合向量列表
    """
    C, T = X_matrix.shape

    X_fft = fft(X_matrix, axis=1)
    freqs = fftfreq(T, 1 / fs)
    half_T = T // 2

    u_list = []
    phi_list = []
    omega_list = []

    lambda_fft = np.zeros_like(X_fft, dtype=complex)
    X_u_fft = X_fft.copy()
    orig_energy = np.sum(np.abs(X_fft[:, :half_T]) ** 2)

    total_steps = max_K * max_iter

    for k in range(max_K):
        u_fft_curr = np.zeros(T, dtype=complex)
        phi_curr = np.random.rand(C, 1) + 1j * np.zeros((C, 1))
        phi_curr /= np.linalg.norm(phi_curr)

        valid_idx = (freqs > 0.05) & (freqs < fs / 2)
        if np.any(valid_idx):
            mean_residual_spec = np.mean(np.abs(X_u_fft[:, valid_idx]), axis=0)
            omega_curr = freqs[valid_idx][np.argmax(mean_residual_spec)]
        else:
            omega_curr = 0.2

        alpha = alpha_min

        for it in range(max_iter):
            u_fft_old = u_fft_curr.copy()

            # 更新 Joint IMF
            num_part1 = np.dot(phi_curr.conj().T, X_fft + lambda_fft / 2.0).flatten()
            num_part2 = (alpha ** 2) * ((freqs - omega_curr) ** 4) * u_fft_old
            numerator = num_part1 + num_part2

            denom_part1 = 1.0 + (alpha ** 2) * ((freqs - omega_curr) ** 4)
            denom_part2 = 1.0 + 2.0 * alpha * ((freqs - omega_curr) ** 2)

            if len(omega_list) > 0:
                penalty_sum = np.zeros(T)
                for omega_past in omega_list:
                    penalty_sum += 1.0 / ((alpha ** 2) * ((freqs - omega_past) ** 4) + 1e-8)
                denom_part2 += penalty_sum

            u_fft_curr[:half_T] = numerator[:half_T] / (denom_part1[:half_T] * denom_part2[:half_T])
            u_fft_curr[half_T:] = 0.0

            # 更新空间混合向量
            weight_filter = 1.0 / (1.0 + (alpha ** 2) * ((freqs[:half_T] - omega_curr) ** 4))
            phi_num = np.dot(X_fft[:, :half_T] + lambda_fft[:, :half_T] / 2.0,
                             (u_fft_curr[:half_T].conj() * weight_filter))
            phi_curr = np.real(phi_num[:, np.newaxis])
            phi_norm = np.linalg.norm(phi_curr)
            if phi_norm > 1e-8:
                phi_curr /= phi_norm

            # 更新中心频率
            u_power = np.abs(u_fft_curr[:half_T]) ** 2
            if np.sum(u_power) > 1e-8:
                omega_curr = np.sum(freqs[:half_T] * u_power) / np.sum(u_power)

            # 双乘子梯度上升
            error_term = (X_fft[:, :half_T] -
                          np.dot(phi_curr, u_fft_curr[:half_T][np.newaxis, :]) +
                          lambda_fft[:, :half_T] / 2.0) * weight_filter[np.newaxis, :]
            lambda_fft[:, :half_T] = lambda_fft[:, :half_T] + (error_term - lambda_fft[:, :half_T] / 2.0)

            alpha = min(gamma * alpha, alpha_max)

            if progress_callback:
                current_step = k * max_iter + it + 1
                if not progress_callback(current_step, total_steps, k + 1, it + 1):
                    return u_list, phi_list

            u_change = (np.linalg.norm(u_fft_curr[:half_T] - u_fft_old[:half_T]) /
                        (np.linalg.norm(u_fft_old[:half_T]) + 1e-8))
            if u_change < epsilon1 and alpha >= alpha_max:
                if progress_callback:
                    remaining = max_iter - it - 1
                    progress_callback(current_step + remaining, total_steps, k + 1, max_iter)
                break

        u_time = 2.0 * np.real(ifft(u_fft_curr))
        u_list.append(u_time)
        phi_list.append(phi_curr)
        omega_list.append(omega_curr)

        X_u_fft -= np.dot(phi_curr, u_fft_curr[np.newaxis, :])

        residual_energy = np.sum(np.abs(X_u_fft[:, :half_T]) ** 2)
        if (residual_energy / orig_energy) < epsilon2:
            if progress_callback:
                progress_callback(total_steps, total_steps, k + 1, max_iter)
            break

    return u_list, phi_list


def extract_smvmd(frames, fs, progress_callback=None):
    """
    SMVMD 呼吸信号提取入口。

    1. 将空间帧展开为多通道信号
    2. SMVMD 递推分解
    3. SNR 门限多分量重构
    """
    N, Row, Col = frames.shape
    X_multichannel = frames.reshape(N, Row * Col).T

    X_active_list = []
    for c in range(X_multichannel.shape[0]):
        channel_sig = X_multichannel[c, :].astype(np.float32)
        if np.std(channel_sig) > 12.0:
            X_active_list.append(channel_sig - np.mean(channel_sig))

    if len(X_active_list) == 0:
        return np.zeros(N)

    X_shrunk = np.array(X_active_list)
    u_modes, _ = SMVMD_Core(X_shrunk, fs, alpha_min=1.0, alpha_max=3000.0,
                             max_K=3, epsilon2=1e-4,
                             progress_callback=progress_callback)

    reconstructed = np.zeros(N)
    found_any = False
    for comp in u_modes:
        n = len(comp)
        freqs = fftfreq(n, 1 / fs)[:n // 2]
        fft_vals = np.abs(fft(comp))[:n // 2]
        dom_freq = freqs[np.argmax(fft_vals)]
        if 0.1 <= dom_freq <= 0.4:
            snr = calculate_snr(comp, fs)
            if snr >= 3.0:
                reconstructed += comp
                found_any = True

    if not found_any and len(u_modes) > 0:
        return u_modes[0]
    return reconstructed
