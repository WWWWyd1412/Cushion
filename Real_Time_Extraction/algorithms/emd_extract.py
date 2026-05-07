from PyEMD import EMD
from .base import get_spatial_sum, select_best_component

def extract_respiration(frames, fs=10.0):
    """经验模态分解提取呼吸"""
    signal_1d = get_spatial_sum(frames, pressure_threshold=100)
    emd = EMD()
    imfs = emd(signal_1d) # 分解得到本征模态函数[cite: 22]
    return select_best_component(imfs, fs)