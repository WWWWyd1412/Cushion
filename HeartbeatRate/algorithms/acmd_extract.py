import numpy as np
from .base import get_dual_roi_mean, select_best_component


def ACMD_Core(signal, fs, max_components=6, tol=1e-4):
    """
    针对心脉信号定制的自适应线性调频模态分解 (ACMD)
    """
    components = []
    residual = signal.copy()
    orig_energy = np.sum(signal ** 2)
    
    for i in range(max_components):
        # 1. 寻找当前残差的主频率，忽略小于 0.75 Hz 的极低频/呼吸残留成分
        n = len(residual)
        fft_vals = np.abs(np.fft.fft(residual))[:n // 2]
        freqs = np.fft.fftfreq(n, 1/fs)[:n // 2]
        
        # 限制在心搏段找初始主频，避免受残留低频呼吸波影响
        valid_idx = (freqs >= 0.75) & (freqs <= 2.5)
        if np.any(valid_idx):
            filtered_fft = fft_vals[valid_idx]
            filtered_freqs = freqs[valid_idx]
            init_freq = filtered_freqs[np.argmax(filtered_fft)]
        else:
            init_freq = 1.2  # 默认心跳中心频率为 1.2 Hz (72 BPM)

        # 2. 时频脊线解调与拟合
        t = np.arange(n) / fs
        c = np.cos(2 * np.pi * init_freq * t)
        s = np.sin(2 * np.pi * init_freq * t)
        
        # 最小二乘自适应滤波拟合
        comp_i = c * (np.dot(residual, c) / (np.dot(c, c) + 1e-6)) + s * (np.dot(residual, s) / (np.dot(s, s) + 1e-6))
        
        # 3. 剥离并记录
        residual -= comp_i
        components.append(comp_i)
        
        # 4. 终止条件检查
        current_mape = np.sum(residual ** 2) / (orig_energy + 1e-12)
        if current_mape < tol:
            break
            
    return np.array(components), residual


def extract_heartbeat(frames, fs):
    """基于 ACMD 的自适应心率信号提取"""
    signal_1d = get_dual_roi_mean(frames, window_size=5)
    if len(signal_1d) == 0:
        return np.zeros(100)
    
    # 交流去趋势
    signal_1d = signal_1d - np.mean(signal_1d)
    
    components, _ = ACMD_Core(signal_1d, fs, max_components=6, tol=0.0001)
    
    # 仅选择心搏生理频段 (0.8 - 2.2Hz) 内能量最大的单一分量
    return select_best_component(components, fs)
