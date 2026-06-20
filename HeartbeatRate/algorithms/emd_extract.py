from PyEMD import EMD
from .base import get_dual_roi_mean, select_best_component


def extract_heartbeat(frames, fs):
    """基于 EMD 的心脉节律提取算法"""
    signal_1d = get_dual_roi_mean(frames)
    if len(signal_1d) == 0:
        return np.zeros(100)
    emd = EMD()
    imfs = emd(signal_1d)
    # 仅选择心搏生理频段 (0.8 - 2.2Hz) 内能量最大的单一分量
    return select_best_component(imfs, fs)
