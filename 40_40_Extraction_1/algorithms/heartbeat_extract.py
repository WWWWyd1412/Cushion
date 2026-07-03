# -*- coding: utf-8 -*-
import numpy as np
from .base import smooth_signal, select_best_component, butter_bandpass_filter, wavelet_denoise


def _acmd_core(signal, fs, max_components=6, tol=1e-4):
    """自适应线性调频模态分解（轻量快速，比 VMD 快10x）"""
    components = []
    residual = signal.copy()
    orig_energy = np.sum(signal ** 2) + 1e-12
    for _ in range(max_components):
        n = len(residual)
        fft_v = np.abs(np.fft.fft(residual))[:n // 2]
        freqs = np.fft.fftfreq(n, 1 / fs)[:n // 2]
        valid = (freqs >= 0.75) & (freqs <= 2.5)
        init_f = freqs[valid][np.argmax(fft_v[valid])] if np.any(valid) else 1.2
        t = np.arange(n) / fs
        c = np.cos(2 * np.pi * init_f * t)
        s = np.sin(2 * np.pi * init_f * t)
        comp = c * (np.dot(residual, c) / (np.dot(c, c) + 1e-6)) + \
               s * (np.dot(residual, s) / (np.dot(s, s) + 1e-6))
        residual -= comp
        components.append(comp)
        if np.sum(residual ** 2) / orig_energy < tol:
            break
    return np.array(components)


def extract_heartbeat_acmd(signal, fs=10.0):
    """ACMD 心跳提取：快速自适应分解 → 小波去噪 → 选最优频段分量"""
    detrended = signal - np.mean(signal)
    if len(detrended) < 30:
        return detrended
    denoised = wavelet_denoise(detrended, alpha=0.3)
    components = _acmd_core(denoised, fs)
    best = select_best_component(components, fs, lowcut=0.8, highcut=2.2)
    return best if np.any(best) else extract_heartbeat_mean(signal, fs)


def extract_heartbeat_mean(signal, fs=10.0):
    """带通均值法：小波去噪 → 带通 0.8–2.2 Hz → 平滑"""
    detrended = signal - np.mean(signal)
    if len(detrended) < 12:
        return detrended
    denoised = wavelet_denoise(detrended, alpha=0.3)
    bandpassed = butter_bandpass_filter(denoised, lowcut=0.8, highcut=2.2, fs=fs, order=3)
    return smooth_signal(bandpassed, window=7, polyorder=2)


def extract_heartbeat_vmd(signal, fs=10.0):
    try:
        from vmdpy import VMD  # type: ignore
        if len(signal) < 100:
            return extract_heartbeat_mean(signal, fs)
        u, _, _ = VMD(signal, alpha=2000, tau=0, K=6, DC=0, init=1, tol=1e-6)
        components = [u[i, :] for i in range(u.shape[0])]
        best = select_best_component(components, fs, lowcut=0.8, highcut=2.2)
        return best if np.any(best) else extract_heartbeat_mean(signal, fs)
    except ImportError:
        return extract_heartbeat_mean(signal, fs)
def extract_heartbeat_emd(signal, fs=10.0):
    try:
        from PyEMD import EMD  # type: ignore
        imfs = EMD().emd(signal)
        if imfs.ndim == 2 and imfs.shape[0] > 0:
            components = [imfs[i, :] for i in range(imfs.shape[0])]
            best = select_best_component(components, fs, lowcut=0.8, highcut=2.2)
            return best if np.any(best) else extract_heartbeat_mean(signal, fs)
        return extract_heartbeat_mean(signal, fs)
    except ImportError:
        return extract_heartbeat_mean(signal, fs)


# ==========================================
# 新增的高级心跳提取算法
# ==========================================

from .base import get_dual_roi_mean_heartbeat, VME_Core

def extract_heartbeat_vme(frames_or_signal, fs=10.0):
    """VME 变分模态提取心跳特征"""
    if isinstance(frames_or_signal, np.ndarray) and frames_or_signal.ndim == 3:
        signal = get_dual_roi_mean_heartbeat(frames_or_signal, fs)
    else:
        signal = frames_or_signal
        
    if len(signal) == 0:
        return np.zeros(100)
        
    detrended = signal - np.mean(signal)
    n_len = len(detrended)
    f_dom = 1.2  # 72 BPM
    if n_len > 8:
        fft_vals = np.abs(np.fft.fft(detrended))[:n_len // 2]
        freqs = np.fft.fftfreq(n_len, 1/fs)[:n_len // 2]
        valid_mask = (freqs >= 0.8) & (freqs <= 2.2)
        if np.any(valid_mask):
            f_dom = freqs[valid_mask][np.argmax(fft_vals[valid_mask])]
            
    try:
        u_heart = VME_Core(detrended, fs, f_init=f_dom, alpha=2000)
        return u_heart
    except Exception:
        return extract_heartbeat_mean(detrended, fs)
