import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
import pywt  # 需要安装 PyWavelets


class Preprocessor:
    def __init__(self, deadzone=35, absolute_max=5000):
        self.deadzone = deadzone
        self.absolute_max = absolute_max

    def process_frame(self, frame):
        """
        对单帧 32x32 矩阵进行空间增强
        """
        f = frame.astype(np.float32)
        # 1. 过滤传感器死区
        f[f < self.deadzone] = 0
        # 2. 空间中值滤波剔除孤立噪点
        f = median_filter(f, size=3)
        # 3. 空间高斯平滑使受力边缘圆润
        f = gaussian_filter(f, sigma=0.5)
        return f


def wavelet_denoise_signal(signal, wavelet='db4', level=3):
    """
    对降维后的一维呼吸信号进行小波自适应去噪
    """
    if len(signal) < 10: return signal

    # 信号分解
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # 计算噪声标准差（基于高频分量）
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    # 计算通用阈值
    threshold = np.sqrt(2 * np.log(len(signal))) * sigma

    # 对细节系数进行软阈值处理
    coeffs_denoised = [pywt.threshold(c, threshold, mode='soft') if i > 0 else c
                       for i, c in enumerate(coeffs)]

    # 信号重构
    return pywt.waverec(coeffs_denoised, wavelet)[:len(signal)]