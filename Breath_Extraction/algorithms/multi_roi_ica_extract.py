import numpy as np
from .base import get_multi_roi_signals, fuse_signals_ica


def extract_respiration(frames, fs):
    """
    轻量级直接多 ROI FastICA 呼吸信号提取算法
    直接提取 4 个核心象限 ROI，通过独立成分分析 (ICA) 融合为单通道生理呼吸信号。
    """
    # 1. 提取多 ROI 信号
    multi_signals = get_multi_roi_signals(frames, num_rois=4, window_size=5)
    if len(multi_signals) == 0:
        return np.zeros(len(frames))
        
    # 2. 直接进行 FastICA 时空源分离与相位对齐
    fused_signal = fuse_signals_ica(multi_signals, fs)
    
    return fused_signal
