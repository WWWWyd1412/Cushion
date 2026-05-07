from vmdpy import VMD
from .base import get_spatial_sum, select_best_component


def extract_respiration(frames, fs=10.0, K=5, alpha=2000):
    """
    K: 分解模态数，通常为 5
    alpha: 带宽限制，通常为 2000
    """
    # 1. 空间降维：提取每帧中 >100 的活跃受力点均值
    # 注意：get_spatial_sum 内部实现了你要求的每一帧独立阈值筛选逻辑
    signal_1d = get_spatial_sum(frames, pressure_threshold=100)

    if len(signal_1d) < 100:
        return signal_1d

    # 2. VMD 分解
    # tau=0 (无直流分量限制), DC=0 (不保留DC), init=1 (中心频率初始化)
    u, u_hat, omega = VMD(signal_1d,
                          alpha=alpha,
                          tau=0,
                          K=K,
                          DC=0,
                          init=1,
                          tol=1e-7)

    # 3. 筛选主频在 0.1-0.5Hz 的呼吸分量
    best_breath_comp = select_best_component(u, fs)

    return best_breath_comp