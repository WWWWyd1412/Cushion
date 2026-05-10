import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, savgol_filter, welch
import pywt


def wavelet_denoise(signal, alpha=0.5):
    """【预处理】自适应小波去噪：去除原始信号中的高频毛刺"""
    if len(signal) < 16: return signal
    coeffs = pywt.wavedec(signal, 'db4', level=3)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr = alpha * np.sqrt(2.0 * np.log(len(signal))) * sigma
    coeffs_dn = [coeffs[0]] + [pywt.threshold(c, thr, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs_dn, 'db4')[:len(signal)]


def calculate_snr(signal, fs=10.0, band=(0.1, 0.4)):
    """【质量评估】计算分量在呼吸频段内的信噪比 (SNR)"""
    f, psd = welch(signal, fs, nperseg=min(len(signal), 256))
    idx_band = np.logical_and(f >= band[0], f <= band[1])
    if not np.any(idx_band): return -10.0
    signal_pwr = np.sum(psd[idx_band])
    noise_pwr = np.sum(psd[~idx_band])
    return 10 * np.log10(signal_pwr / noise_pwr) if noise_pwr > 0 else 20.0

def get_dual_roi_mean(frames, window_size=5):
    """【入口统一】ROI 提取 + 立即执行小波预处理"""
    offset = window_size // 2
    signal_1d = []
    for f in frames:
        l_zone = f[:, :16]
        r_zone = f[:, 16:]
        l_idx = np.unravel_index(np.argmax(l_zone), l_zone.shape)
        r_idx = np.unravel_index(np.argmax(r_zone), r_zone.shape)
        r_idx = (r_idx[0], r_idx[1] + 16)

        def get_roi_mean(matrix, center_idx):
            r, c = center_idx
            r_s, r_e = max(0, r - offset), min(31, r + offset + 1)
            c_s, c_e = max(0, c - offset), min(31, c + offset + 1)
            roi = matrix[r_s:r_e, c_s:c_e]
            return np.mean(roi) if roi.size > 0 else 0

        signal_1d.append((get_roi_mean(f, l_idx) + get_roi_mean(f, r_idx)) / 2)

    return wavelet_denoise(np.array(signal_1d))

def reconstruct_multicomponent_with_snr(components, fs, snr_threshold=3.0):
    """
    【核心变更】全员入选逻辑：
    不再只选一个分量，而是叠加所有符合频率(0.1-0.4Hz)且 SNR 达标的分量。
    """
    if components is None or len(components) == 0:
        return np.zeros(100)

    reconstructed_signal = np.zeros_like(components[0])
    found_any = False

    for comp in components:
        n = len(comp)
        freqs = fftfreq(n, 1/fs)[:n // 2]
        fft_vals = np.abs(fft(comp))[:n // 2]
        dom_freq = freqs[np.argmax(fft_vals)]

        # 1. 频率判定：0.1 ~ 0.4 Hz
        if 0.1 <= dom_freq <= 0.4:
            # 2. 质量判定：SNR 必须大于阈值
            snr = calculate_snr(comp, fs)
            if snr >= snr_threshold:
                reconstructed_signal += comp
                found_any = True

    # 保底逻辑：如果没有任何分量达标，返回能量最大的原始分量
    return reconstructed_signal if found_any else components[0]



def calculate_bpm_fpr(signal, fs, k1=0.3):
    """
    新方法频率计算：基于文献 VMD-FPR 的 TH1 阈值识别
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