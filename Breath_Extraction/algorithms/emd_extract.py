from PyEMD import EMD
from .base import get_dual_roi_mean, reconstruct_multicomponent_with_snr

def extract_respiration(frames, fs):
    signal_1d = get_dual_roi_mean(frames)
    emd = EMD()
    imfs = emd(signal_1d)
    # 同样应用全员入选逻辑
    return reconstruct_multicomponent_with_snr(imfs, fs)