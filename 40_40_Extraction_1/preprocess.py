# -*- coding: utf-8 -*-
"""
40x40 压力阵列空间预处理模块
包含：死区过滤、中值滤波、高斯平滑
"""

import numpy as np
from scipy.ndimage import median_filter, gaussian_filter


class Preprocessor:
    """
    40x40 压力帧空间预处理器
    """

    def __init__(self, deadzone=30):
        """
        Parameters:
            deadzone: 死区阈值，低于此值的像素置零
        """
        self.deadzone = deadzone

    def process_frame(self, frame):
        """
        对单帧 40x40 压力矩阵进行空间预处理

        Parameters:
            frame: (40, 40) ndarray

        Returns:
            processed: (40, 40) float64 ndarray
        """
        # 1. 转换为浮点
        f = frame.astype(np.float64)

        # 2. 裁剪到合理范围
        f = np.clip(f, 0, 2000)

        # 3. 死区阈值过滤
        f[f < self.deadzone] = 0

        # 4. 3×3 中值滤波（去椒盐噪声）
        f = median_filter(f, size=3)

        # 5. 高斯平滑（sigma=0.5）
        f = gaussian_filter(f, sigma=0.5)

        return f


def wavelet_denoise_signal(signal, wavelet="db4", level=3, alpha=0.5):
    """
    小波去噪（1D 信号），可用于后处理

    Parameters:
        signal: 1D 输入信号
        wavelet: 小波基名称
        level: 分解层数
        alpha: 阈值系数

    Returns:
        去噪后的信号
    """
    try:
        import pywt

        if len(signal) < 16:
            return signal
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        thr = alpha * np.sqrt(2.0 * np.log(len(signal))) * sigma
        coeffs_dn = [coeffs[0]] + [
            pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]
        ]
        return pywt.waverec(coeffs_dn, wavelet)[: len(signal)]
    except ImportError:
        return signal
