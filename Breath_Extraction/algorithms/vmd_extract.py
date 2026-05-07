from vmdpy import VMD
from .base import get_dual_roi_mean, select_best_component


def extract_respiration(frames, fs, K=5, alpha=2000):
    signal_1d = get_dual_roi_mean(frames, window_size=5)
    u, _, _ = VMD(signal_1d, alpha=alpha, tau=0, K=K, DC=0, init=1, tol=1e-7)
    return select_best_component(u, fs)