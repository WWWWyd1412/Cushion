"""
心跳分析专用配置
================
集中管理心跳 (BCG) 频段的所有硬编码参数。
"""

from dataclasses import dataclass


@dataclass
class HeartbeatConfig:
    """心跳 (BCG) 信号提取的默认参数配置"""

    # --- 频带 ---
    FREQ_BAND: tuple = (0.8, 2.2)        # 心跳生理频带 (Hz) — 48~132 BPM
    SNR_BAND: tuple = (0.8, 2.2)          # SNR 计算频带 (Hz)

    # --- 小波去噪 ---
    WAVELET_ALPHA: float = 0.3            # 更温和以保留 J 峰

    # --- Savitzky-Golay 平滑 ---
    SAVGOL_WINDOW: int = 7                # 小窗口，避免平滑掉 1-2Hz 的心跳细节
    SAVGOL_ORDER: int = 2

    # --- VMD 参数 ---
    VMD_K: int = 6                        # 心跳模态数
    VMD_ALPHA: int = 2000

    # --- BPM 峰值检测 ---
    BPM_MIN_DIST_SEC: float = 0.4         # 峰值最小间距 (秒) — 对应最高 150 BPM
    BPM_PROMINENCE_RATIO: float = 0.15    # Prominence 系数

    # --- SNR 重构 ---
    SNR_THRESHOLD: float = 3.0

    # --- 预处理 ---
    GAUSSIAN_SIGMA: float = 0.0           # 心跳不启用空间高斯滤波
    USE_VME_BASELINE: bool = True         # 心跳启用 VME 基线漂移去除

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in vars(cls).items()
                if not k.startswith('_') and k.isupper()}
