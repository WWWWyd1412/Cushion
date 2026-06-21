"""
统一信号处理工具
================
参数化的滤波、去噪、SNR计算、平滑函数。

合并了 Breath_Extraction 和 HeartbeatRate 的 base.py 中重复的函数:
    - butter_bandpass_filter  (仅频带参数不同)
    - wavelet_denoise         (仅 alpha 参数不同)
    - calculate_snr           (仅频带参数不同)
    - smooth_signal           (仅窗口/阶数不同)
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, savgol_filter, welch, butter, filtfilt
import pywt
import warnings
from sklearn.exceptions import ConvergenceWarning

# 忽略 FastICA 无法收敛的 ConvergenceWarning 警告
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------------------------------------------------------
# 滤波
# ---------------------------------------------------------------------------

def butter_bandpass_filter(data, lowcut, highcut, fs=10.0, order=3):
    """
    Butterworth 带通滤波器。

    Parameters
    ----------
    data : ndarray
        输入信号。
    lowcut : float
        低频截止 (Hz)。
    highcut : float
        高频截止 (Hz)。
    fs : float
        采样率。
    order : int
        滤波器阶数。

    Returns
    -------
    ndarray — 滤波后信号。
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # 限制高频在有效 Nyquist 范围内
    if high >= 1.0:
        high = 0.99
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


# ---------------------------------------------------------------------------
# 去噪
# ---------------------------------------------------------------------------

def wavelet_denoise(signal, alpha=0.5, wavelet='db4', level=3):
    """
    自适应小波软阈值去噪。

    Parameters
    ----------
    signal : ndarray
        输入信号。
    alpha : float
        阈值系数。呼吸用 0.5~1.2，心跳用 0.3 (更温和保留 J 峰)。
    wavelet : str
        小波基名称。
    level : int
        分解层数。

    Returns
    -------
    ndarray — 去噪后信号。
    """
    if len(signal) < 16:
        return signal

    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    if sigma < 1e-12:
        return signal

    thr = alpha * np.sqrt(2.0 * np.log(len(signal))) * sigma
    coeffs_dn = [coeffs[0]] + [
        pywt.threshold(c, thr, mode='soft') for c in coeffs[1:]
    ]
    return pywt.waverec(coeffs_dn, wavelet)[:len(signal)]


# ---------------------------------------------------------------------------
# 质量评估
# ---------------------------------------------------------------------------

def calculate_snr(signal, fs=10.0, band=(0.1, 0.4)):
    """
    计算指定频段内的信噪比 (Signal-to-Noise Ratio)。

    Parameters
    ----------
    signal : ndarray
        输入信号。
    fs : float
        采样率。
    band : tuple
        (low, high) 目标频带 (Hz)。

    Returns
    -------
    float — SNR (dB)，若无法计算则返回 -10.0。
    """
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


# ---------------------------------------------------------------------------
# 平滑
# ---------------------------------------------------------------------------

def smooth_signal(signal, window_size=41, polyorder=3):
    """
    Savitzky-Golay 平滑滤波。

    Parameters
    ----------
    signal : ndarray
        输入信号。
    window_size : int
        窗口大小 (必须为奇数)。呼吸 41，心跳 7。
    polyorder : int
        多项式阶数。呼吸 3，心跳 2。

    Returns
    -------
    ndarray — 平滑后信号。
    """
    if len(signal) < window_size:
        return signal
    return savgol_filter(signal, window_size, polyorder)


# ---------------------------------------------------------------------------
# 峰值检测 (BPM 计算)
# ---------------------------------------------------------------------------

def calculate_bpm_peak(signal, fs=10.0, min_dist_sec=1.5, prominence_ratio=0.5):
    """
    基于 prominence 的峰值检测 BPM 计算。

    Parameters
    ----------
    signal : ndarray
    fs : float
    min_dist_sec : float
        峰值最小间距 (秒)。呼吸 1.5s，心跳 0.4s。
    prominence_ratio : float
        基于标准差的 prominence 系数。呼吸 0.5，心跳 0.15。

    Returns
    -------
    float — BPM 值。
    """
    std_val = np.std(signal)
    if std_val < 1e-6:
        return 0.0

    prom = max(std_val * prominence_ratio, 1e-3)
    peaks, _ = find_peaks(signal, distance=int(fs * min_dist_sec), prominence=prom)

    if len(peaks) < 2:
        return 0.0
    return (60 * fs) / np.mean(np.diff(peaks))


def calculate_bpm_fpr(signal, fs=10.0, min_dist_sec=1.5, k1=0.3):
    """
    基于 FPR (Feature Point Recognition) TH1 阈值的 BPM 计算。

    Parameters
    ----------
    signal : ndarray
    fs : float
    min_dist_sec : float
        峰值最小间距 (秒)。呼吸 1.5s，心跳 0.4s。
    k1 : float
        TH1 阈值系数。

    Returns
    -------
    float — BPM 值。
    """
    min_dist = int(fs * min_dist_sec)
    peaks, _ = find_peaks(signal, distance=min_dist)
    troughs, _ = find_peaks(-signal, distance=min_dist)

    if len(peaks) < 2 or len(troughs) < 1:
        # 兜底：回退到无距离限制
        peaks, _ = find_peaks(signal)
        troughs, _ = find_peaks(-signal)

    if len(peaks) < 2 or len(troughs) < 1:
        return 0.0

    c_max = np.max(signal[peaks])
    t_min = np.min(signal[troughs])
    th1 = k1 * abs(c_max - t_min)

    # 筛选满足 TH1 条件的主波
    main_waves = [p for p in peaks if (signal[p] - t_min) > th1]
    if len(main_waves) < 2:
        return 0.0

    avg_interval = np.mean(np.diff(main_waves)) / fs
    return 60 / avg_interval
