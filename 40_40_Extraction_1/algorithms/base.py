# -*- coding: utf-8 -*-
"""
40x40 压力阵列实时信号处理公共算法
包含：空间求和降维、平滑、BPM 计算、分量筛选、带通滤波
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt


def wavelet_denoise(signal, alpha=0.3):
    """db4 小波软阈值去噪，保留生理频段细节"""
    try:
        import pywt
        if len(signal) < 16:
            return signal
        coeffs = pywt.wavedec(signal, 'db4', level=3)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        if sigma < 1e-12:
            return signal
        thr = alpha * np.sqrt(2.0 * np.log(len(signal))) * sigma
        coeffs_dn = [coeffs[0]] + [pywt.threshold(c, thr, mode='soft') for c in coeffs[1:]]
        return pywt.waverec(coeffs_dn, 'db4')[:len(signal)]
    except ImportError:
        return signal


def calculate_bpm_fpr(signal, fs=10.0, min_dist_s=1.2, k1=0.3):
    """
    FPR（特征点识别）BPM：比 prominence 方法更鲁棒，
    用幅值阈值 TH1 筛选主波，抑制噪声毛刺。
    min_dist_s: 心跳用 0.4s，呼吸用 1.5s
    """
    min_dist = max(1, int(fs * min_dist_s))
    peaks, _ = find_peaks(signal, distance=min_dist)
    troughs, _ = find_peaks(-signal, distance=min_dist)
    if len(peaks) < 2 or len(troughs) < 1:
        return 0.0
    c_max = np.max(signal[peaks])
    t_min = np.min(signal[troughs])
    th1 = k1 * abs(c_max - t_min)
    main_waves = [p for p in peaks if (signal[p] - t_min) > th1]
    if len(main_waves) < 2:
        return 0.0
    return (60.0 * fs) / np.mean(np.diff(main_waves))


def get_spatial_sum(frames, pressure_threshold=100):
    """
    将 40x40 帧序列降维为 1D 信号。
    每帧取大于阈值的压力点均值，并去均值中心化。

    Parameters:
        frames: (N, 40, 40) 或 list of (40, 40) ndarray
        pressure_threshold: 压力阈值，低于此值的像素不参与计算

    Returns:
        signal_1d: (N,) 去均值后的 1D 信号
    """
    if isinstance(frames, list):
        frames = np.array(frames)
    if frames.ndim == 2:
        frames = frames[np.newaxis, :, :]

    signal_1d = []
    for f in frames:
        active_points = f[f > pressure_threshold]
        if active_points.size > 0:
            signal_1d.append(np.mean(active_points))
        else:
            non_zero = f[f > 0]
            signal_1d.append(np.mean(non_zero) if non_zero.size > 0 else 0)

    signal_1d = np.array(signal_1d)
    return signal_1d - np.mean(signal_1d)


def smooth_signal(signal, window=11, polyorder=3):
    """
    Savitzky-Golay 平滑滤波

    Parameters:
        signal: 1D 输入信号
        window: 窗口长度（必须为奇数）
        polyorder: 多项式阶数

    Returns:
        平滑后的信号
    """
    if len(signal) < window:
        return signal
    return savgol_filter(signal, window, polyorder)


def calculate_bpm(signal, fs=10.0, min_distance=1.2):
    """
    通过波峰检测计算每分钟次数（BPM）

    Parameters:
        signal: 1D 信号
        fs: 采样率 (Hz)
        min_distance: 波峰间最小间隔 (秒)

    Returns:
        bpm: 每分钟次数，信号过弱时返回 0.0
    """
    if np.max(signal) - np.min(signal) < 1e-6:
        return 0.0
    peaks, _ = find_peaks(
        signal,
        distance=int(fs * min_distance),
        prominence=(np.max(signal) - np.min(signal)) * 0.2,
    )
    if len(peaks) < 2:
        return 0.0
    avg_interval = np.mean(np.diff(peaks))
    return (60.0 * fs) / avg_interval


def select_best_component(components, fs, lowcut=0.1, highcut=0.5):
    """
    从多个信号分量中筛选目标频段内能量最大的分量

    Parameters:
        components: 分量列表，每个分量是 1D ndarray
        fs: 采样率 (Hz)
        lowcut: 目标频段下限 (Hz)
        highcut: 目标频段上限 (Hz)

    Returns:
        best_comp: 最优分量，未找到时返回零数组
    """
    if components is None or len(components) == 0:
        return np.zeros(200)

    best_comp = None
    max_energy = -1.0
    for comp in components:
        n = len(comp)
        if n == 0:
            continue
        freqs = fftfreq(n, 1.0 / fs)[: n // 2]
        fft_vals = np.abs(fft(comp))[: n // 2]
        dom_freq = freqs[np.argmax(fft_vals)]

        if lowcut <= dom_freq <= highcut:
            energy = np.sqrt(np.mean(comp**2))
            if energy > max_energy:
                max_energy = energy
                best_comp = comp

    if best_comp is not None:
        return best_comp
    return np.zeros_like(components[0]) if len(components) > 0 else np.zeros(200)


def butter_bandpass_filter(data, lowcut=0.1, highcut=0.5, fs=10.0, order=3):
    """
    巴特沃斯带通滤波器

    Parameters:
        data: 1D 输入信号
        lowcut: 低截止频率 (Hz)
        highcut: 高截止频率 (Hz)
        fs: 采样率 (Hz)
        order: 滤波器阶数

    Returns:
        滤波后的信号
    """
    if len(data) < 30:
        return data  # 信号太短，无法有效滤波
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if high >= 1.0:
        high = 0.99
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


# ==========================================
# 新增的高级算法辅助函数
# ==========================================

def VME_Core(signal, fs, f_init=0.25, alpha=1000, tol=1e-6, max_iter=150):
    """
    变分模态提取 (VME) 核心实现，包含镜像延拓
    """
    T = len(signal)
    if T < 10:
        return np.zeros_like(signal)
    
    pad_len = min(T // 2, 100)
    left_pad = signal[1:pad_len+1][::-1]
    right_pad = signal[-pad_len-1:-1][::-1]
    padded_signal = np.concatenate((left_pad, signal, right_pad))
    T_pad = len(padded_signal)
    
    f_fft = np.fft.fft(padded_signal)
    half_T = T_pad // 2
    f_fft_analytic = np.zeros_like(f_fft, dtype=complex)
    f_fft_analytic[0] = f_fft[0]
    f_fft_analytic[1:half_T] = 2.0 * f_fft[1:half_T]
    if T_pad % 2 == 0:
        f_fft_analytic[half_T] = f_fft[half_T]
        
    freqs = np.fft.fftfreq(T_pad, 1/fs)
    
    u_fft = np.zeros(T_pad, dtype=complex)
    lambda_fft = np.zeros(T_pad, dtype=complex)
    omega_d = f_init
    tau = 0.1
    
    for it in range(max_iter):
        u_fft_old = u_fft.copy()
        diff = freqs[:half_T+1] - omega_d
        diff2 = diff**2
        diff4 = diff2**2
        
        num = f_fft_analytic[:half_T+1] + (alpha**2) * diff4 * u_fft_old[:half_T+1] + lambda_fft[:half_T+1] / 2.0
        den = (1.0 + (alpha**2) * diff4) * (1.0 + 2.0 * alpha * diff2)
        
        u_fft[:half_T+1] = num / (den + 1e-12)
        u_fft[half_T+1:] = 0.0
        
        u_power = np.abs(u_fft[:half_T+1])**2
        sum_power = np.sum(u_power)
        if sum_power > 1e-12:
            omega_d = np.sum(freqs[:half_T+1] * u_power) / sum_power
            
        error = (f_fft_analytic[:half_T+1] - u_fft[:half_T+1]) / (1.0 + (alpha**2) * diff4 + 1e-12)
        lambda_fft[:half_T+1] = lambda_fft[:half_T+1] + tau * error
        
        if it > 5:
            change = np.linalg.norm(u_fft[:half_T+1] - u_fft_old[:half_T+1]) / (np.linalg.norm(u_fft_old[:half_T+1]) + 1e-12)
            if change < tol:
                break
                
    u_time_padded = np.real(np.fft.ifft(u_fft))
    return u_time_padded[pad_len : pad_len + T]


def get_dual_roi_mean_breath(frames, fs=10.0, window_size=5):
    """自适应锁定左右受力中心，并滤波提取 1D 呼吸信号"""
    if len(frames) == 0:
        return np.array([])
    
    offset = window_size // 2
    signal_1d = []
    
    stable_mean_frame = None
    trigger_threshold = 120
    stability_window = min(20, len(frames))
    
    for i in range(len(frames) - stability_window + 1):
        if np.max(frames[i]) > trigger_threshold:
            sub_series = frames[i : i + stability_window]
            frame_means = [np.mean(f) for f in sub_series]
            stability_score = np.std(frame_means)
            if stability_score < 5.0:
                stable_mean_frame = np.mean(sub_series, axis=0)
                break
                
    if stable_mean_frame is None:
        stable_mean_frame = np.mean(frames, axis=0)
        
    h, w = stable_mean_frame.shape
    mid = w // 2
    l_zone = stable_mean_frame[:, :mid]
    r_zone = stable_mean_frame[:, mid:]
    
    l_idx = np.unravel_index(np.argmax(l_zone), l_zone.shape)
    r_idx = np.unravel_index(np.argmax(r_zone), r_zone.shape)
    r_idx = (r_idx[0], r_idx[1] + mid)
    
    for f in frames:
        def get_roi_mean(matrix, center_idx):
            r, c = center_idx
            r_s, r_e = max(0, r - offset), min(h - 1, r + offset + 1)
            c_s, c_e = max(0, c - offset), min(w - 1, c + offset + 1)
            roi = matrix[r_s:r_e, c_s:c_e]
            return np.mean(roi) if roi.size > 0 else 0
            
        signal_1d.append((get_roi_mean(f, l_idx) + get_roi_mean(f, r_idx)) / 2)
        
    sig_wavelet = wavelet_denoise(np.array(signal_1d), alpha=1.2)
    sig_bandpass = butter_bandpass_filter(sig_wavelet, lowcut=0.1, highcut=0.5, fs=fs, order=3)
    return sig_bandpass


def get_dual_roi_mean_heartbeat(frames, fs=10.0, window_size=5):
    """自适应锁定左右中心，提取心跳 1D 信号并使用 VME 剥除呼吸基线漂移"""
    if len(frames) == 0:
        return np.array([])
        
    offset = window_size // 2
    signal_1d = []
    
    stable_mean_frame = None
    trigger_threshold = 120
    stability_window = min(20, len(frames))
    
    for i in range(len(frames) - stability_window + 1):
        if np.max(frames[i]) > trigger_threshold:
            sub_series = frames[i : i + stability_window]
            frame_means = [np.mean(f) for f in sub_series]
            stability_score = np.std(frame_means)
            if stability_score < 5.0:
                stable_mean_frame = np.mean(sub_series, axis=0)
                break
                
    if stable_mean_frame is None:
        stable_mean_frame = np.mean(frames, axis=0)
        
    h, w = stable_mean_frame.shape
    mid = w // 2
    l_zone = stable_mean_frame[:, :mid]
    r_zone = stable_mean_frame[:, mid:]
    
    l_idx = np.unravel_index(np.argmax(l_zone), l_zone.shape)
    r_idx = np.unravel_index(np.argmax(r_zone), r_zone.shape)
    r_idx = (r_idx[0], r_idx[1] + mid)
    
    for f in frames:
        def get_roi_mean(matrix, center_idx):
            r, c = center_idx
            r_s, r_e = max(0, r - offset), min(h - 1, r + offset + 1)
            c_s, c_e = max(0, c - offset), min(w - 1, c + offset + 1)
            roi = matrix[r_s:r_e, c_s:c_e]
            return np.mean(roi) if roi.size > 0 else 0
            
        signal_1d.append((get_roi_mean(f, l_idx) + get_roi_mean(f, r_idx)) / 2)
        
    sig_raw_np = np.array(signal_1d)
    sig_demeaned = sig_raw_np - np.mean(sig_raw_np)
    
    n_len = len(sig_demeaned)
    if n_len > 8:
        fft_vals = np.abs(np.fft.fft(sig_demeaned))[:n_len // 2]
        freqs = np.fft.fftfreq(n_len, 1/fs)[:n_len // 2]
        valid_mask = (freqs >= 0.8) & (freqs <= 2.2)
        f_heart = freqs[valid_mask][np.argmax(fft_vals[valid_mask])] if np.any(valid_mask) else 1.2
    else:
        f_heart = 1.2
        
    if f_heart <= 1.25:
        alpha_bd = 1000.0 * np.exp(1.09 * ((f_heart - 1.25) / -0.5) ** 2)
    else:
        alpha_bd = 1000.0
        
    u_BD = VME_Core(sig_demeaned, fs=fs, f_init=0.25, alpha=alpha_bd)
    sig_bd_removed = sig_demeaned - u_BD
    
    sig_wavelet = wavelet_denoise(sig_bd_removed, alpha=0.3)
    sig_bandpass = butter_bandpass_filter(sig_wavelet, lowcut=0.8, highcut=2.2, fs=fs, order=3)
    return sig_bandpass


def get_multi_roi_signals_40x40(frames, fs=10.0, num_rois=4, window_size=5):
    """自适应提取多 ROI 通道信号"""
    if len(frames) == 0:
        return np.array([])
        
    offset = window_size // 2
    
    stable_mean_frame = None
    trigger_threshold = 120
    stability_window = min(20, len(frames))
    
    for i in range(len(frames) - stability_window + 1):
        if np.max(frames[i]) > trigger_threshold:
            sub_series = frames[i : i + stability_window]
            frame_means = [np.mean(f) for f in sub_series]
            stability_score = np.std(frame_means)
            if stability_score < 5.0:
                stable_mean_frame = np.mean(sub_series, axis=0)
                break
                
    if stable_mean_frame is None:
        stable_mean_frame = np.mean(frames, axis=0)
        
    h, w = stable_mean_frame.shape
    centers = []
    
    if num_rois == 4:
        h_half = h // 2
        w_half = w // 2
        quadrants = [
            (0, h_half, 0, w_half, 0, 0),
            (0, h_half, w_half, w, 0, w_half),
            (h_half, h, 0, w_half, h_half, 0),
            (h_half, h, w_half, w, h_half, w_half)
        ]
        for r_s, r_e, c_s, c_e, r_off, c_off in quadrants:
            zone = stable_mean_frame[r_s:r_e, c_s:c_e]
            idx = np.unravel_index(np.argmax(zone), zone.shape)
            centers.append((idx[0] + r_off, idx[1] + c_off))
    elif num_rois == 2:
        w_half = w // 2
        l_zone = stable_mean_frame[:, :w_half]
        r_zone = stable_mean_frame[:, w_half:]
        l_idx = np.unravel_index(np.argmax(l_zone), l_zone.shape)
        r_idx = np.unravel_index(np.argmax(r_zone), r_zone.shape)
        centers = [l_idx, (r_idx[0], r_idx[1] + w_half)]
    else:
        flat_idx = np.argsort(stable_mean_frame.ravel())[::-1]
        for idx_flat in flat_idx:
            r, c = np.unravel_index(idx_flat, stable_mean_frame.shape)
            too_close = False
            for cr, cc in centers:
                if abs(cr - r) < 6 and abs(cc - c) < 6:
                    too_close = True
                    break
            if not too_close:
                centers.append((r, c))
                if len(centers) == num_rois:
                    break
        while len(centers) < num_rois:
            centers.append((h // 2, w // 2))
            
    num_frames = len(frames)
    signals = np.zeros((num_rois, num_frames))
    
    for k, center in enumerate(centers):
        r, c = center
        r_s, r_e = max(0, r - offset), min(h - 1, r + offset + 1)
        c_s, c_e = max(0, c - offset), min(w - 1, c + offset + 1)
        
        for i, f in enumerate(frames):
            roi = f[r_s:r_e, c_s:c_e]
            signals[k, i] = np.mean(roi) if roi.size > 0 else 0.0
            
    processed_signals = np.zeros((num_rois, num_frames))
    for k in range(num_rois):
        sig = signals[k, :]
        sig_dn = wavelet_denoise(sig, alpha=1.2)
        sig_bp = butter_bandpass_filter(sig_dn, lowcut=0.1, highcut=0.5, fs=fs, order=3)
        processed_signals[k, :] = sig_bp
        
    return processed_signals


def calculate_snr(signal, fs=10.0, band=(0.1, 0.4)):
    """计算信号在 target 生理频带内的信噪比 (SNR)"""
    from scipy.signal import welch
    nperseg = min(len(signal), 256)
    if nperseg < 8:
        return -10.0
    f, psd = welch(signal, fs, nperseg=nperseg)
    idx_band = np.logical_and(f >= band[0], f <= band[1])
    if not np.any(idx_band):
        return -10.0
    signal_pwr = np.sum(psd[idx_band])
    noise_pwr = np.sum(psd[~idx_band])
    return 10 * np.log10(signal_pwr / noise_pwr) if noise_pwr > 0 else 20.0


def reconstruct_multicomponent_with_snr(components, fs, snr_threshold=3.0, band=(0.1, 0.4)):
    """多分量 SNR 重构"""
    if components is None or len(components) == 0:
        return np.zeros(100)

    reconstructed_signal = np.zeros_like(components[0])
    found_any = False

    for comp in components:
        n = len(comp)
        freqs = fftfreq(n, 1/fs)[:n // 2]
        fft_vals = np.abs(fft(comp))[:n // 2]
        dom_freq = freqs[np.argmax(fft_vals)]

        if band[0] <= dom_freq <= band[1]:
            snr = calculate_snr(comp, fs, band=band)
            if snr >= snr_threshold:
                reconstructed_signal += comp
                found_any = True

    return reconstructed_signal if found_any else components[0]


def fuse_signals_ica(multi_channel_signals, fs=10.0, band=(0.1, 0.4)):
    """多通道 FastICA 盲源分离融合"""
    try:
        from sklearn.decomposition import FastICA
    except ImportError:
        return fuse_signals_pca(multi_channel_signals)
        
    M, T = multi_channel_signals.shape
    if M == 1:
        return multi_channel_signals[0]
        
    X = multi_channel_signals.T
    n_comps = min(3, M)
    
    ica = FastICA(n_components=n_comps, random_state=42, max_iter=1000, tol=1e-3)
    try:
        sources = ica.fit_transform(X)
    except Exception as e:
        print(f"[ICA Warning] FastICA 异常: {e}. 自动回退至 PCA 融合。")
        return fuse_signals_pca(multi_channel_signals)
        
    best_source = None
    max_snr = -999.0
    
    for k in range(n_comps):
        comp = sources[:, k]
        n = len(comp)
        freqs = fftfreq(n, 1/fs)[:n // 2]
        fft_vals = np.abs(fft(comp))[:n // 2]
        dom_freq = freqs[np.argmax(fft_vals)]
        
        if band[0] <= dom_freq <= band[1]:
            snr = calculate_snr(comp, fs, band=band)
            if snr > max_snr:
                max_snr = snr
                best_source = comp
                
    if best_source is not None:
        channel_mean = np.mean(multi_channel_signals, axis=0)
        if np.dot(best_source, channel_mean) < 0:
            best_source = -best_source
        return best_source
    else:
        return sources[:, 0]


def fuse_signals_pca(multi_channel_signals):
    """PCA 信号融合"""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return multi_channel_signals[0]
        
    X = multi_channel_signals.T
    pca = PCA(n_components=1, random_state=42)
    source = pca.fit_transform(X).flatten()
    
    channel_mean = np.mean(multi_channel_signals, axis=0)
    if np.dot(source, channel_mean) < 0:
        source = -source
    return source
