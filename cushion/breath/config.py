"""
呼吸分析专用配置
================
集中管理呼吸频段的所有硬编码参数。
"""

from dataclasses import dataclass


@dataclass
class BreathConfig:
    """呼吸信号提取的默认参数配置"""

    # --- 频带 ---
    FREQ_BAND: tuple = (0.1, 0.5)       # 呼吸生理频带 (Hz)
    SNR_BAND: tuple = (0.1, 0.4)         # SNR 计算频带 (Hz)

    # --- 小波去噪 ---
    WAVELET_ALPHA: float = 0.5           # 阈值系数

    # --- Savitzky-Golay 平滑 ---
    SAVGOL_WINDOW: int = 41              # 窗口大小
    SAVGOL_ORDER: int = 3                # 多项式阶数

    # --- VMD 参数 ---
    VMD_K: int = 5                       # 模态数
    VMD_ALPHA: int = 2000                # 带宽约束

    # --- BPM 峰值检测 ---
    BPM_MIN_DIST_SEC: float = 1.5        # 峰值最小间距 (秒) — 对应最高 40 BPM
    BPM_PROMINENCE_RATIO: float = 0.5    # Prominence 系数

    # --- SNR 重构 ---
    SNR_THRESHOLD: float = 3.0           # SNR 门限 (dB)

    # --- 预处理 ---
    GAUSSIAN_SIGMA: float = 0.8          # 空间高斯滤波 sigma

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in vars(cls).items()
                if not k.startswith('_') and k.isupper()}
