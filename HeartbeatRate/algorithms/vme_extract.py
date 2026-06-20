import numpy as np
from .base import get_dual_roi_mean, VME_Core


def extract_heartbeat(frames, fs):
    """
    基于 VME (变分模态提取) 的心跳节律提取算法
    直接定位信号在生理区间 [0.8, 2.2] Hz 内的主频，并提取该单一变分模态
    """
    signal_1d = get_dual_roi_mean(frames, window_size=5)
    if len(signal_1d) == 0:
        return np.zeros(100)
    
    # 交流去趋势
    signal_1d = signal_1d - np.mean(signal_1d)
    
    # 估计主频
    n_len = len(signal_1d)
    f_dom = 1.2  # 默认 72 BPM
    if n_len > 8:
        fft_vals = np.abs(np.fft.fft(signal_1d))[:n_len // 2]
        freqs = np.fft.fftfreq(n_len, 1/fs)[:n_len // 2]
        valid_mask = (freqs >= 0.8) & (freqs <= 2.2)
        if np.any(valid_mask):
            f_dom = freqs[valid_mask][np.argmax(fft_vals[valid_mask])]
            
    # 运行 VME 提取心跳主频对应的变分模态分量 (设置 alpha = 2000 作为平衡参数)
    u_heart = VME_Core(signal_1d, fs, f_init=f_dom, alpha=2000)
    return u_heart
