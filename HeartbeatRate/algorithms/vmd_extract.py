from vmdpy import VMD
import numpy as np
from .base import get_dual_roi_mean, select_best_component


def extract_heartbeat(frames, fs, K=6, alpha=2000):
    """基于 VMD 的心脉节律提取算法"""
    signal_1d = get_dual_roi_mean(frames, window_size=5)
    if len(signal_1d) == 0:
        return np.zeros(100)
    # 调用 VMD 进行分解
    u, _, _ = VMD(signal_1d, alpha=alpha, tau=0, K=K, DC=0, init=1, tol=1e-7)
    # 仅选择心搏生理频段 (0.8 - 2.2Hz) 内能量最大的单一分量
    return select_best_component(u, fs)
