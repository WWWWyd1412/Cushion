from PyEMD import EMD
from .base import get_dual_roi_mean, select_best_component


def extract_respiration(frames, fs):
    signal_1d = get_dual_roi_mean(frames,window_size=5)
    emd = EMD()
    imfs = emd(signal_1d)
    return select_best_component(imfs, fs)