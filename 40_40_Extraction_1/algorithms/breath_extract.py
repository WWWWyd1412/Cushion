# -*- coding: utf-8 -*-
import numpy as np
from .base import smooth_signal, select_best_component, wavelet_denoise


def extract_breath_mean(signal):
    """均值法：小波去噪 + Savitzky-Golay 平滑"""
    detrended = signal - np.mean(signal)
    denoised = wavelet_denoise(detrended, alpha=0.5)
    return smooth_signal(denoised, window=11, polyorder=3)


def extract_breath_vmd(signal, fs=10.0):
    try:
        from vmdpy import VMD  # type: ignore
        if len(signal) < 100:
            return extract_breath_mean(signal)
        u, _, _ = VMD(signal, alpha=2000, tau=0, K=5, DC=0, init=1, tol=1e-6)
        components = [u[i, :] for i in range(u.shape[0])]
        best = select_best_component(components, fs, lowcut=0.1, highcut=0.5)
        return best if np.any(best) else extract_breath_mean(signal)
    except ImportError:
        return extract_breath_mean(signal)


def extract_breath_emd(signal, fs=10.0):
    try:
        from PyEMD import EMD  # type: ignore
        imfs = EMD().emd(signal)
        if imfs.ndim == 2 and imfs.shape[0] > 0:
            components = [imfs[i, :] for i in range(imfs.shape[0])]
            best = select_best_component(components, fs, lowcut=0.1, highcut=0.5)
            return best if np.any(best) else extract_breath_mean(signal)
        return extract_breath_mean(signal)
    except ImportError:
        return extract_breath_mean(signal)


def extract_breath_afd(signal, fs=10.0):
    n = len(signal)
    if n < 50:
        return extract_breath_mean(signal)
    t = np.arange(n) / fs
    candidates = [np.real(np.exp(2j * np.pi * f * t)) for f in np.linspace(0.1, 0.5, 20)]
    best = select_best_component(candidates, fs, lowcut=0.1, highcut=0.5)
    return best if np.any(best) else extract_breath_mean(signal)


# ==========================================
# 新增的高级呼吸提取算法
# ==========================================

from .base import (
    get_dual_roi_mean_breath,
    get_multi_roi_signals_40x40,
    fuse_signals_ica,
    reconstruct_multicomponent_with_snr,
    calculate_snr
)
from scipy.signal import hilbert
from scipy.fft import fft, ifft, fftfreq

def extract_breath_vmd_mape(frames_or_signal, fs=10.0):
    """VMD-MAPE 呼吸信号提取"""
    if isinstance(frames_or_signal, np.ndarray) and frames_or_signal.ndim == 3:
        signal = get_dual_roi_mean_breath(frames_or_signal, fs)
    else:
        signal = frames_or_signal
        
    if len(signal) < 100:
        return extract_breath_mean(signal)
        
    try:
        from vmdpy import VMD
        mapes = []
        k_range = range(2, 11) 
        best_u = None

        for k in k_range:
            u, _, _ = VMD(signal, alpha=2000, tau=0, K=k, DC=0, init=1, tol=1e-7)
            res = signal - np.sum(u, axis=0)
            mape = np.sum(res ** 2) / (np.sum(signal ** 2) + 1e-12)
            mapes.append(mape)
            if len(mapes) > 1 and mapes[-1] > mapes[-2]:
                break
            best_u = u
            
        if best_u is None:
            return extract_breath_mean(signal)
            
        return reconstruct_multicomponent_with_snr(best_u, fs, snr_threshold=3.0, band=(0.1, 0.4))
    except Exception:
        return extract_breath_mean(signal)


def envelope_entropy(sig):
    analytic = hilbert(sig)
    envelope = np.abs(analytic)
    env_sum = np.sum(envelope)
    if env_sum < 1e-12:
        return 1e10
    env_norm = envelope / env_sum
    env_norm = env_norm[env_norm > 1e-12]
    if len(env_norm) == 0:
        return 1e10
    return -np.sum(env_norm * np.log(env_norm))


def _goa_fitness(params, signal, fs):
    K = int(np.clip(round(params[0]), 2, 10))
    alpha = int(np.clip(params[1], 500, 5000))
    try:
        from vmdpy import VMD
        u, _, omega = VMD(signal, alpha=alpha, tau=0, K=K, DC=0, init=1, tol=1e-7)
        if any(np.any(np.isnan(comp)) for comp in u):
            return 1e10
        avg_entropy = np.mean([envelope_entropy(comp) for comp in u])
        omega_final = omega[-1, :]
        freq_penalty = 0.0
        if len(omega_final) > 1:
            center_freqs_hz = np.sort(omega_final * fs / (2 * np.pi))
            freq_diffs = np.diff(center_freqs_hz)
            freq_penalty = np.sum(np.maximum(0, 0.3 - freq_diffs))
        snr_bonus = 0.0
        for comp in u:
            snr_val = calculate_snr(comp, fs, band=(0.1, 0.4))
            if snr_val > 0:
                snr_bonus += snr_val
        return avg_entropy + 0.15 * freq_penalty - 0.005 * snr_bonus
    except Exception:
        return 1e10


def goa_optimize_light(signal, fs):
    pop_size = 4
    max_iter = 3
    lb = np.array([2.0, 500.0])
    ub = np.array([10.0, 5000.0])
    dim = 2
    
    np.random.seed(42)
    population = np.zeros((pop_size, dim))
    population[:, 0] = np.random.randint(int(lb[0]), int(ub[0]) + 1, pop_size)
    population[:, 1] = np.random.uniform(lb[1], ub[1], pop_size)
    
    fitness = np.array([_goa_fitness(population[i], signal, fs) for i in range(pop_size)])
    best_idx = np.argmin(fitness)
    best_position = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    cmax, cmin = 1.0, 0.00004
    for iteration in range(max_iter):
        c = cmax - iteration * (cmax - cmin) / max_iter
        new_population = np.zeros_like(population)
        for i in range(pop_size):
            S_i = np.zeros(dim)
            for j in range(pop_size):
                if i != j:
                    dist = np.linalg.norm(population[i] - population[j])
                    if dist < 1e-8:
                        dist = 1e-8
                    f_attr, l_attr = 0.5, 1.5
                    s_r = f_attr * np.exp(-dist / l_attr) - np.exp(-dist)
                    direction = (population[j] - population[i]) / dist
                    S_i += s_r * direction
            new_population[i] = c * S_i + best_position
        new_population[:, 0] = np.clip(np.round(new_population[:, 0]), lb[0], ub[0])
        new_population[:, 1] = np.clip(new_population[:, 1], lb[1], ub[1])
        new_population[0] = best_position.copy()
        population = new_population
        fitness = np.array([_goa_fitness(population[i], signal, fs) for i in range(pop_size)])
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_fitness:
            best_fitness = fitness[min_idx]
            best_position = population[min_idx].copy()
            
    K_opt = int(np.clip(round(best_position[0]), 2, 10))
    alpha_opt = int(np.clip(best_position[1], 500, 5000))
    return K_opt, alpha_opt


def extract_breath_goa_vmd(frames_or_signal, fs=10.0):
    """GOA-VMD 自适应呼吸信号提取 (轻量实时版)"""
    if isinstance(frames_or_signal, np.ndarray) and frames_or_signal.ndim == 3:
        signal = get_dual_roi_mean_breath(frames_or_signal, fs)
    else:
        signal = frames_or_signal
        
    if len(signal) < 100:
        return extract_breath_mean(signal)
        
    try:
        from vmdpy import VMD
        K_opt, alpha_opt = goa_optimize_light(signal, fs)
        u, _, _ = VMD(signal, alpha=alpha_opt, tau=0, K=K_opt, DC=0, init=1, tol=1e-7)
        return reconstruct_multicomponent_with_snr(u, fs, snr_threshold=3.0, band=(0.1, 0.4))
    except Exception:
        return extract_breath_mean(signal)


def SMVMD_Core_light(X_matrix, fs, alpha_min=1.0, alpha_max=3000.0, gamma=1.414,
                     epsilon1=1e-7, epsilon2=1e-5, max_K=3, max_iter=100):
    C, T = X_matrix.shape
    X_fft = fft(X_matrix, axis=1)
    freqs = fftfreq(T, 1/fs)
    half_T = T // 2
    u_list, phi_list, omega_list = [], [], []
    lambda_fft = np.zeros_like(X_fft, dtype=complex)
    X_u_fft = X_fft.copy()
    orig_energy = np.sum(np.abs(X_fft[:, :half_T])**2)
    
    for k in range(max_K):
        u_fft_curr = np.zeros(T, dtype=complex)
        phi_curr = np.random.rand(C, 1) + 1j * np.zeros((C, 1))
        phi_curr /= np.linalg.norm(phi_curr)
        
        valid_idx = (freqs > 0.05) & (freqs < fs/2)
        if np.any(valid_idx):
            mean_residual_spec = np.mean(np.abs(X_u_fft[:, valid_idx]), axis=0)
            omega_curr = freqs[valid_idx][np.argmax(mean_residual_spec)]
        else:
            omega_curr = 0.2
            
        alpha = alpha_min
        for it in range(max_iter):
            u_fft_old = u_fft_curr.copy()
            num_part1 = np.dot(phi_curr.conj().T, X_fft + lambda_fft / 2.0).flatten()
            num_part2 = (alpha**2) * ((freqs - omega_curr)**4) * u_fft_old
            numerator = num_part1 + num_part2
            
            denom_part1 = 1.0 + (alpha**2) * ((freqs - omega_curr)**4)
            denom_part2 = 1.0 + 2.0 * alpha * ((freqs - omega_curr)**2)
            if len(omega_list) > 0:
                penalty_sum = np.zeros(T)
                for omega_past in omega_list:
                    penalty_sum += 1.0 / ((alpha**2) * ((freqs - omega_past)**4) + 1e-8)
                denom_part2 += penalty_sum
                
            u_fft_curr[:half_T] = numerator[:half_T] / (denom_part1[:half_T] * denom_part2[:half_T])
            u_fft_curr[half_T:] = 0.0
            
            weight_filter = 1.0 / (1.0 + (alpha**2) * ((freqs[:half_T] - omega_curr)**4))
            phi_num = np.dot(X_fft[:, :half_T] + lambda_fft[:, :half_T] / 2.0, 
                             (u_fft_curr[:half_T].conj() * weight_filter))
            phi_curr = np.real(phi_num[:, np.newaxis])
            phi_norm = np.linalg.norm(phi_curr)
            if phi_norm > 1e-8:
                phi_curr /= phi_norm
                
            u_power = np.abs(u_fft_curr[:half_T])**2
            if np.sum(u_power) > 1e-8:
                omega_curr = np.sum(freqs[:half_T] * u_power) / np.sum(u_power)
                
            error_term = (X_fft[:, :half_T] - np.dot(phi_curr, u_fft_curr[:half_T][np.newaxis, :]) + lambda_fft[:, :half_T] / 2.0) * weight_filter[np.newaxis, :]
            lambda_fft[:, :half_T] = lambda_fft[:, :half_T] + (error_term - lambda_fft[:, :half_T] / 2.0)
            alpha = min(gamma * alpha, alpha_max)
            
            u_change = np.linalg.norm(u_fft_curr[:half_T] - u_fft_old[:half_T]) / (np.linalg.norm(u_fft_old[:half_T]) + 1e-8)
            if u_change < epsilon1 and alpha >= alpha_max:
                break
                
        u_time = 2.0 * np.real(ifft(u_fft_curr))
        u_list.append(u_time)
        phi_list.append(phi_curr)
        omega_list.append(omega_curr)
        
        X_u_fft -= np.dot(phi_curr, u_fft_curr[np.newaxis, :])
        residual_energy = np.sum(np.abs(X_u_fft[:, :half_T])**2)
        if (residual_energy / (orig_energy + 1e-12)) < epsilon2:
            break
            
    return u_list


def extract_breath_smvmd(frames, fs=10.0):
    """SMVMD 自适应多通道呼吸提取 (限制在前10个最活跃通道以保证实时性能)"""
    if not isinstance(frames, np.ndarray) or frames.ndim != 3:
        return extract_breath_vmd(frames, fs)
        
    N, Row, Col = frames.shape
    X_multichannel = frames.reshape(N, Row * Col).T
    
    stds = np.std(X_multichannel, axis=1)
    active_indices = np.where(stds > 12.0)[0]
    if len(active_indices) == 0:
        return np.zeros(N)
        
    sorted_active_indices = active_indices[np.argsort(stds[active_indices])[::-1]]
    top_indices = sorted_active_indices[:10]
    
    X_active_list = [X_multichannel[idx, :].astype(np.float32) - np.mean(X_multichannel[idx, :]) for idx in top_indices]
    X_shrunk = np.array(X_active_list)
    
    try:
        u_modes = SMVMD_Core_light(X_shrunk, fs, alpha_min=1.0, alpha_max=3000.0, max_K=3, epsilon2=1e-4)
        return reconstruct_multicomponent_with_snr(np.array(u_modes), fs, snr_threshold=3.0, band=(0.1, 0.4))
    except Exception:
        return extract_breath_mean(np.mean(frames, axis=(1, 2)))


def mvmd_core(X, alpha=2000, tau=0, K=4, DC=0, init=1, tol=1e-7, max_iter=150):
    M, T = X.shape
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
    
    omega = np.zeros((max_iter, K))
    if init == 1:
        for k in range(K):
            omega[0, k] = 0.5 * k / K
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
            denom = 1.0 + 2.0 * alpha * (freqs_half - omega[it, k])**2
            for m in range(M):
                numerator = X_fft_half[m, :] - sum_other[m, :] - lambda_fft[m, :] / 2.0
                u_fft[k, m, :] = numerator / denom
                
            if not (DC == 1 and k == 0):
                power_spectrum_sum = np.sum(np.abs(u_fft[k, :, :])**2, axis=0)
                denom_freq = np.sum(power_spectrum_sum)
                omega[it + 1, k] = np.sum(freqs_half * power_spectrum_sum) / (denom_freq + 1e-12) if denom_freq > 1e-12 else omega[it, k]
            else:
                omega[it + 1, k] = 0.0
                
        lambda_fft += tau * (X_fft_half - np.sum(u_fft, axis=0))
        
        diff_sum = 0.0
        norm_sum = 0.0
        for k in range(K):
            diff_sum += np.sum(np.abs(u_fft[k] - u_fft_old[k])**2)
            norm_sum += np.sum(np.abs(u_fft_old[k])**2)
            
        if diff_sum / (norm_sum + 1e-12) < tol:
            converged = True
        it += 1
        
    u_fft_full = np.zeros((K, M, T_padded), dtype=complex)
    u_fft_full[:, :, :half_len] = u_fft
    
    u_time = np.zeros((K, M, T))
    for k in range(K):
        for m in range(M):
            analytic = ifft(u_fft_full[k, m, :])
            u_time[k, m, :] = 2.0 * np.real(analytic[half_T : half_T + T])
            
    return u_time


def extract_breath_mvmd(frames, fs=10.0, K=4, alpha=2000):
    """MVMD 多通道呼吸信号提取 (提取4个核心象限受力区域)"""
    if not isinstance(frames, np.ndarray) or frames.ndim != 3:
        return extract_breath_vmd(frames, fs)
        
    multi_signals = get_multi_roi_signals_40x40(frames, fs=fs, num_rois=4, window_size=5)
    if len(multi_signals) == 0:
        return np.zeros(len(frames))
        
    try:
        u_time = mvmd_core(multi_signals, alpha=alpha, tau=0, K=K, DC=0, init=1, tol=1e-7)
        fused_components = []
        for k in range(K):
            fused_k = fuse_signals_ica(u_time[k], fs)
            fused_components.append(fused_k)
        return reconstruct_multicomponent_with_snr(np.array(fused_components), fs, snr_threshold=3.0, band=(0.1, 0.4))
    except Exception:
        return extract_breath_mean(np.mean(frames, axis=(1, 2)))


def extract_breath_multi_roi_ica(frames, fs=10.0):
    """Multi-ROI ICA 呼吸信号提取"""
    if not isinstance(frames, np.ndarray) or frames.ndim != 3:
        return extract_breath_mean(frames)
        
    multi_signals = get_multi_roi_signals_40x40(frames, fs=fs, num_rois=4, window_size=5)
    if len(multi_signals) == 0:
        return np.zeros(len(frames))
        
    return fuse_signals_ica(multi_signals, fs)


def extract_breath_acmd(signal, fs=10.0):
    """ACMD 呼吸信号提取：快速自适应调频模态分解"""
    if isinstance(signal, np.ndarray) and signal.ndim == 3:
        signal = np.mean(signal, axis=(1, 2))
        
    detrended = signal - np.mean(signal)
    if len(detrended) < 30:
        return detrended
        
    from .base import wavelet_denoise, select_best_component
    denoised = wavelet_denoise(detrended, alpha=0.3)
    
    components = []
    residual = denoised.copy()
    orig_energy = np.sum(denoised ** 2) + 1e-12
    max_components = 6
    tol = 1e-4
    
    for _ in range(max_components):
        n = len(residual)
        fft_v = np.abs(np.fft.fft(residual))[:n // 2]
        freqs = np.fft.fftfreq(n, 1 / fs)[:n // 2]
        # 呼吸特征频段：0.08 到 0.6 Hz
        valid = (freqs >= 0.08) & (freqs <= 0.6)
        init_f = freqs[valid][np.argmax(fft_v[valid])] if np.any(valid) else 0.25
        t = np.arange(n) / fs
        c = np.cos(2 * np.pi * init_f * t)
        s = np.sin(2 * np.pi * init_f * t)
        comp = c * (np.dot(residual, c) / (np.dot(c, c) + 1e-6)) + \
               s * (np.dot(residual, s) / (np.dot(s, s) + 1e-6))
        residual -= comp
        components.append(comp)
        if np.sum(residual ** 2) / orig_energy < tol:
            break
            
    components = np.array(components)
    best = select_best_component(components, fs, lowcut=0.1, highcut=0.5)
    return best if np.any(best) else extract_breath_mean(signal, fs)
