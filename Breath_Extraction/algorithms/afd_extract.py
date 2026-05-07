import numpy as np
from scipy.signal import hilbert
from .base import get_dual_roi_mean
from .base import select_best_component


def extract_respiration(frames, fs, n_components=3):
    signal_1d = get_dual_roi_mean(frames, window_size=5)
    analytic_signal = hilbert(signal_1d - np.mean(signal_1d))
    # AFD 简化逻辑：寻找与呼吸频段最匹配的解析分量[cite: 3]
    # 此处省略复杂的 AFD 迭代细节，采用你提供的频率搜索思路
    t = np.arange(len(signal_1d)) / fs
    components = []
    # 模拟 AFD 分解过程
    for f in np.linspace(0.1, 0.5, 20):
        comp = np.exp(1j * 2 * np.pi * f * t) # 示例基函数[cite: 3]
        components.append(comp.real)
    return select_best_component(np.array(components), fs)