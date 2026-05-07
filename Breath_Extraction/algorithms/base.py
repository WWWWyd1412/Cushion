import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, savgol_filter


def get_dual_roi_mean(frames, window_size=5):
    """
    新方法选点：动态追踪左右臀部压力中心提取 5x5 区域均值
    """
    offset = window_size // 2
    signal_1d = []
    for f in frames:
        # 寻找左右区域最大值
        left_zone = f[:, :16]
        right_zone = f[:, 16:]
        l_idx = np.unravel_index(np.argmax(left_zone), left_zone.shape)
        r_idx_raw = np.unravel_index(np.argmax(right_zone), right_zone.shape)
        r_idx = (r_idx_raw[0], r_idx_raw[1] + 16)

        def get_roi_mean(matrix, center_idx):
            r, c = center_idx
            r_s, r_e = max(0, r - offset), min(31, r + offset + 1)
            c_s, c_e = max(0, c - offset), min(31, c + offset + 1)
            roi = matrix[r_s:r_e, c_s:c_e]
            return np.mean(roi) if roi.size > 0 else 0

        signal_1d.append((get_roi_mean(f, l_idx) + get_roi_mean(f, r_idx)) / 2)
    signal_1d = np.array(signal_1d)
    return signal_1d - np.mean(signal_1d)


def calculate_bpm_fpr(signal, fs, k1=0.3):
    """
    新方法频率计算：基于文献 VMD-FPR 的 TH1 阈值识别[cite: 11, 16]
    """
    peaks, _ = find_peaks(signal)
    troughs, _ = find_peaks(-signal)
    if len(peaks) < 2 or len(troughs) < 1: return 0.0

    c_max = np.max(signal[peaks])
    t_min = np.min(signal[troughs])
    th1 = k1 * abs(c_max - t_min) # 计算 TH1 阈值

    # 筛选满足 TH1 条件的主波
    main_waves = [p for p in peaks if (signal[p] - t_min) > th1]
    if len(main_waves) < 2: return 0.0

    avg_interval = np.mean(np.diff(main_waves)) / fs
    return 60 / avg_interval


def calculate_bpm(signal, fs=10.0):
    """原有方法：基于 prominence 的峰值检测"""
    peaks, _ = find_peaks(signal, distance=int(fs * 1.2),
                         prominence=(np.max(signal) - np.min(signal)) * 0.2)
    if len(peaks) < 2: return 0.0
    return (60 * fs) / np.mean(np.diff(peaks))


def smooth_respiration_signal(signal, window_size=41, polyorder=3):
    if len(signal) < window_size: return signal
    return savgol_filter(signal, window_size, polyorder)


def select_best_component(components, fs):
    """筛选 0.1-0.5Hz 呼吸频段内能量最大的分量"""
    # --- 关键修复：检查 components 是否为空 ---
    if components is None or len(components) == 0:
        return np.zeros(200)  # 返回一个与 buffer 长度一致的全零数组

    best_comp = None
    max_energy = -1
    for comp in components:
        n = len(comp)
        if n == 0: continue  # 跳过空分量

        freqs = fftfreq(n, 1 / fs)[:n // 2]
        fft_vals = np.abs(fft(comp))[:n // 2]
        dom_freq = freqs[np.argmax(fft_vals)]

        if 0.1 <= dom_freq <= 0.5:
            energy = np.sqrt(np.mean(comp ** 2))
            if energy > max_energy:
                max_energy = energy
                best_comp = comp

    # --- 关键修复：确保返回值安全 ---
    if best_comp is not None:
        return best_comp
    else:
        # 如果没有找到符合频段的分量，返回第一个分量的全零版本或直接返回全零
        return np.zeros_like(components[0]) if len(components) > 0 else np.zeros(200)