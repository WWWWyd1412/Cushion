import numpy as np
from .base import get_spatial_sum, select_best_component

def extract_respiration(frames, fs=10.0):
    """简化版 AFD：在呼吸频段内搜索最匹配的基函数"""
    signal_1d = get_spatial_sum(frames, pressure_threshold=100)
    t = np.arange(len(signal_1d)) / fs
    components = []
    # 在 0.1 到 0.5Hz 范围内模拟 AFD 的频率基函数搜索
    for f in np.linspace(0.1, 0.5, 20):
        comp = np.exp(1j * 2 * np.pi * f * t)
        components.append(comp.real)
    return select_best_component(np.array(components), fs)