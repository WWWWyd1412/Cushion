"""
算法共享基础函数
================
合并 Breath_Extraction / HeartbeatRate 中 base.py 的重复代码。

所有频带相关的参数从调用方传入 (来自 cushion.breath.config 或 cushion.heartbeat.config)，
实现同一套代码服务呼吸和心跳两个领域。
"""

import numpy as np
from scipy.fft import fft, fftfreq
from cushion.core.signal_utils import (
    butter_bandpass_filter,
    wavelet_denoise,
    calculate_snr,
)


# =====================================================================
# 分量选择与重构 (参数化频带)
# =====================================================================

def select_best_component(components, fs, freq_band=(0.1, 0.5)):
    """
    筛选指定频段内能量 (RMS) 最大的分量。

    Parameters
    ----------
    components : ndarray, shape (K, N)
    fs : float
    freq_band : tuple (low, high)
    """
    if components is None or len(components) == 0:
        return np.zeros(200)

    best_comp = None
    max_energy = -1
    for comp in components:
        n = len(comp)
        if n == 0:
            continue

        freqs = fftfreq(n, 1 / fs)[:n // 2]
        fft_vals = np.abs(fft(comp))[:n // 2]
        dom_freq = freqs[np.argmax(fft_vals)]

        if freq_band[0] <= dom_freq <= freq_band[1]:
            energy = np.sqrt(np.mean(comp ** 2))
            if energy > max_energy:
                max_energy = energy
                best_comp = comp

    if best_comp is not None:
        return best_comp
    return np.zeros_like(components[0]) if len(components) > 0 else np.zeros(200)


def reconstruct_multicomponent_with_snr(components, fs, freq_band=(0.1, 0.4),
                                         snr_threshold=3.0):
    """
    SNR 门限多分量重构: 叠加所有落入目标频段且 SNR 达标的分量。
    """
    if components is None or len(components) == 0:
        return np.zeros(100)

    reconstructed = np.zeros_like(components[0])
    found_any = False

    for comp in components:
        n = len(comp)
        freqs = fftfreq(n, 1 / fs)[:n // 2]
        fft_vals = np.abs(fft(comp))[:n // 2]
        dom_freq = freqs[np.argmax(fft_vals)]

        if freq_band[0] <= dom_freq <= freq_band[1]:
            snr = calculate_snr(comp, fs, band=freq_band)
            if snr >= snr_threshold:
                reconstructed += comp
                found_any = True

    return reconstructed if found_any else components[0]


def reconstruct_top3_by_energy(components, fs, freq_band=(0.1, 0.4)):
    """
    Top-3 能量自适应重构: 选频段内能量前 3 的分量叠加。
    """
    if components is None or len(components) == 0:
        return np.zeros(100)

    valid = []
    for comp in components:
        n = len(comp)
        freqs = fftfreq(n, 1 / fs)[:n // 2]
        fft_vals = np.abs(fft(comp))[:n // 2]
        dom_freq = freqs[np.argmax(fft_vals)]

        if freq_band[0] <= dom_freq <= freq_band[1]:
            rms = np.sqrt(np.mean(comp ** 2))
            valid.append({'component': comp, 'energy': rms, 'freq': dom_freq})

    if not valid:
        return components[0]

    valid.sort(key=lambda x: x['energy'], reverse=True)
    top_k = min(3, len(valid))

    result = np.zeros_like(components[0])
    for info in valid[:top_k]:
        result += info['component']
    return result


# =====================================================================
# ROI 信号提取 (参数化后处理链)
# =====================================================================

def get_dual_roi_mean(frames, fs=10.0, window_size=5,
                       freq_band=(0.1, 0.5), wavelet_alpha=0.5,
                       use_vme_baseline=False):
    """
    自适应稳定坐姿检测 + 双 ROI (左/右臀部) 1D 信号提取。

    合并了 Breath (呼吸) 和 Heartbeat (心跳) 两个版本的差异:
      - Breath: wavelet -> bandpass
      - Heartbeat: VME基线去除 -> wavelet -> bandpass

    Parameters
    ----------
    frames : ndarray, shape (N, 32, 32)
    fs : float
    window_size : int
    freq_band : tuple (lowcut, highcut) Hz
    wavelet_alpha : float
        小波去噪强度系数。呼吸 ~0.5~1.2，心跳 ~0.3。
    use_vme_baseline : bool
        是否启用 VME 基线漂移去除 (心跳专用)。
    """
    if len(frames) == 0:
        return np.array([])

    offset = window_size // 2
    signal_1d = []

    # --- 自适应稳定坐姿检测 ---
    stable_mean_frame = None
    trigger_threshold = 120
    stability_window = 20

    for i in range(len(frames) - stability_window):
        if np.max(frames[i]) > trigger_threshold:
            sub_series = frames[i:i + stability_window]
            frame_means = [np.mean(f) for f in sub_series]
            stability_score = np.std(frame_means)
            if stability_score < 5.0:
                stable_mean_frame = np.mean(sub_series, axis=0)
                break

    if stable_mean_frame is None:
        stable_mean_frame = np.mean(frames, axis=0)

    # --- 锁定左右 ROI 中心 ---
    l_zone = stable_mean_frame[:, :16]
    r_zone = stable_mean_frame[:, 16:]
    l_idx = np.unravel_index(np.argmax(l_zone), l_zone.shape)
    r_idx = np.unravel_index(np.argmax(r_zone), r_zone.shape)
    r_idx = (r_idx[0], r_idx[1] + 16)

    # --- 提取 1D 信号 ---
    for f in frames:
        def _roi_mean(matrix, center_idx):
            r, c = center_idx
            r_s, r_e = max(0, r - offset), min(31, r + offset + 1)
            c_s, c_e = max(0, c - offset), min(31, c + offset + 1)
            roi = matrix[r_s:r_e, c_s:c_e]
            return np.mean(roi) if roi.size > 0 else 0

        signal_1d.append((_roi_mean(f, l_idx) + _roi_mean(f, r_idx)) / 2)

    sig = np.array(signal_1d)
    sig = sig - np.mean(sig)

    # --- 级联后处理链路 ---
    if use_vme_baseline:
        # 心跳: VME 去除基线漂移 (呼吸干扰 ~0.25 Hz)
        from cushion.algorithms.decomposition.vme import _vme_core
        # 自适应平衡因子
        n_len = len(sig)
        if n_len > 8:
            fft_vals = np.abs(np.fft.fft(sig))[:n_len // 2]
            freqs_arr = np.fft.fftfreq(n_len, 1 / fs)[:n_len // 2]
            valid_mask = (freqs_arr >= freq_band[0]) & (freqs_arr <= freq_band[1])
            if np.any(valid_mask):
                f_heart = freqs_arr[valid_mask][np.argmax(fft_vals[valid_mask])]
            else:
                f_heart = freq_band[1] / 2
        else:
            f_heart = freq_band[1] / 2

        if f_heart <= 1.25:
            alpha_bd = 1000.0 * np.exp(1.09 * ((f_heart - 1.25) / -0.5) ** 2)
        else:
            alpha_bd = 1000.0

        u_bd = _vme_core(sig, fs=fs, f_init=0.25, alpha=alpha_bd)
        sig = sig - u_bd

    sig = wavelet_denoise(sig, alpha=wavelet_alpha)
    sig = butter_bandpass_filter(sig, lowcut=freq_band[0], highcut=freq_band[1],
                                  fs=fs, order=3)
    return sig


def get_multi_roi_signals(frames, fs=10.0, num_rois=4, window_size=5,
                           freq_band=(0.1, 0.5), wavelet_alpha=0.5):
    """
    多 ROI 信号提取: 将 32x32 阵列划分为多个区域，每个区域提取 1D 信号。

    Parameters
    ----------
    frames : ndarray, shape (N, 32, 32)
    fs : float
    num_rois : int
        2, 4, 或自定义数量。
    window_size : int
        ROI 窗口半宽。
    freq_band : tuple
    wavelet_alpha : float
    """
    if len(frames) == 0:
        return np.array([])

    offset = window_size // 2

    # 自适应稳定坐姿检测
    stable_mean_frame = None
    trigger_threshold = 120
    stability_window = 20

    for i in range(len(frames) - stability_window):
        if np.max(frames[i]) > trigger_threshold:
            sub_series = frames[i:i + stability_window]
            frame_means = [np.mean(f) for f in sub_series]
            if np.std(frame_means) < 5.0:
                stable_mean_frame = np.mean(sub_series, axis=0)
                break

    if stable_mean_frame is None:
        stable_mean_frame = np.mean(frames, axis=0)

    # 定位 ROI 中心
    centers = []
    if num_rois == 4:
        quadrants = [
            (0, 16, 0, 16, 0, 0),
            (0, 16, 16, 32, 0, 16),
            (16, 32, 0, 16, 16, 0),
            (16, 32, 16, 32, 16, 16),
        ]
        for r_s, r_e, c_s, c_e, r_off, c_off in quadrants:
            zone = stable_mean_frame[r_s:r_e, c_s:c_e]
            idx = np.unravel_index(np.argmax(zone), zone.shape)
            centers.append((idx[0] + r_off, idx[1] + c_off))
    elif num_rois == 2:
        l_zone = stable_mean_frame[:, :16]
        r_zone = stable_mean_frame[:, 16:]
        l_idx = np.unravel_index(np.argmax(l_zone), l_zone.shape)
        r_idx = np.unravel_index(np.argmax(r_zone), r_zone.shape)
        centers = [l_idx, (r_idx[0], r_idx[1] + 16)]
    else:
        flat_idx = np.argsort(stable_mean_frame.ravel())[::-1]
        for idx_flat in flat_idx:
            r, c = np.unravel_index(idx_flat, stable_mean_frame.shape)
            too_close = any(abs(cr - r) < 6 and abs(cc - c) < 6 for cr, cc in centers)
            if not too_close:
                centers.append((r, c))
                if len(centers) == num_rois:
                    break
        while len(centers) < num_rois:
            centers.append((16, 16))

    # 提取各 ROI 的 1D 时序
    n_frames = len(frames)
    signals = np.zeros((num_rois, n_frames))
    for k, (r, c) in enumerate(centers):
        r_s, r_e = max(0, r - offset), min(31, r + offset + 1)
        c_s, c_e = max(0, c - offset), min(31, c + offset + 1)
        for i, f in enumerate(frames):
            roi = f[r_s:r_e, c_s:c_e]
            signals[k, i] = np.mean(roi) if roi.size > 0 else 0.0

    # 每个通道单独去噪 + 带通滤波
    processed = np.zeros_like(signals)
    for k in range(num_rois):
        s = signals[k, :]
        s = wavelet_denoise(s, alpha=wavelet_alpha)
        s = butter_bandpass_filter(s, lowcut=freq_band[0], highcut=freq_band[1],
                                    fs=fs, order=3)
        processed[k, :] = s

    return processed


def get_spatial_sum(frames, pressure_threshold=100):
    """
    简单阈值空间求和 (Real_Time_Extraction 专用)。
    每帧提取大于阈值的像素均值。
    """
    signal_1d = []
    for f in frames:
        active = f[f > pressure_threshold]
        if active.size > 0:
            signal_1d.append(np.mean(active))
        else:
            nonzero = f[f > 0]
            signal_1d.append(np.mean(nonzero) if nonzero.size > 0 else 0)
    signal_1d = np.array(signal_1d)
    return signal_1d - np.mean(signal_1d)
