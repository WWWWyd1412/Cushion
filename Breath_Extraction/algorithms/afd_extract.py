import numpy as np
from scipy.signal import hilbert
from .base import get_dual_roi_mean, reconstruct_multicomponent_with_snr

def extract_respiration(frames, fs, n_components=5):
    """
    升级版 AFD：集成预处理与多组分 SNR 重构
    """
    # 1. 预处理：ROI 提取 + 小波去噪
    signal_1d = get_dual_roi_mean(frames, window_size=5)
    
    # 2. 转换为解析信号 (Hilbert 变换)
    z = hilbert(signal_1d - np.mean(signal_1d))
    t = np.arange(len(z)) / fs
    
    residual = z.copy()
    components = []
    
    # 3. 模拟 AFD 迭代搜索逻辑 (寻找 0.1-0.5Hz 频段内的基函数)
    search_freqs = np.linspace(0.1, 0.5, 50)
    for _ in range(n_components):
        best_comp = None
        max_proj = -1
        
        for f in search_freqs:
            # 生成候选基函数 (r=0.95 确保在单位圆内)
            kernel = np.exp(1j * 2 * np.pi * f * t)
            proj = np.abs(np.vdot(residual, kernel))
            
            if proj > max_proj:
                max_proj = proj
                best_comp = (np.vdot(residual, kernel) / np.vdot(kernel, kernel)) * kernel
        
        if best_comp is not None:
            components.append(np.real(best_comp))
            residual = residual - best_comp
            
    # 4. 全员入选逻辑：叠加所有符合频率和 SNR 条件的 AFD 分量
    return reconstruct_multicomponent_with_snr(np.array(components), fs)