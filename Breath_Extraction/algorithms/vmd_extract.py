from vmdpy import VMD
from .base import get_dual_roi_mean, reconstruct_multicomponent_with_snr

def extract_respiration(frames, fs, K=5, alpha=2000):
    # get_dual_roi_mean 内部已完成小波去噪
    signal_1d = get_dual_roi_mean(frames, window_size=5)
    u, _, _ = VMD(signal_1d, alpha=alpha, tau=0, K=K, DC=0, init=1, tol=1e-7)
    # 变更为多组分重构
    return reconstruct_multicomponent_with_snr(u, fs)