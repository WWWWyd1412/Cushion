import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, savgol_filter # 引入寻峰和平滑库

def get_spatial_sum(frames, pressure_threshold=100):
    """每一帧只提取大于阈值的数值进行分析"""
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

def calculate_bpm(signal, fs=10.0):
    """通过波峰检测计算每分钟呼吸次数"""
    if np.max(signal) - np.min(signal) < 1e-6:
        return 0.0
    # 呼吸间隔需 > 1.2s，显著度需 > 幅度的 20%
    peaks, _ = find_peaks(signal, distance=int(fs * 1.2),
                          prominence=(np.max(signal) - np.min(signal)) * 0.2)
    if len(peaks) < 2:
        return 0.0
    avg_interval = np.mean(np.diff(peaks))
    return (60 * fs) / avg_interval

def smooth_signal(signal, window=11, polyorder=3):
    """使用 Savitzky-Golay 滤波器平滑实时波形"""
    if len(signal) < window:
        return signal
    return savgol_filter(signal, window, polyorder)