"""
统一预处理器
============
参数化的 Preprocessor 类和 clean_dataset 函数，合并了三个模块的差异。

参数差异来源:
    - Breath: sigma=0.8 (高斯滤波), crop 在清洗之后
    - Heartbeat: 无高斯滤波, crop 在清洗之前, fs 参数化
    - RealTime: 无校准系统, sigma=0.5

统一方案: 所有差异通过构造函数参数控制。
"""

import numpy as np
from scipy.ndimage import median_filter, gaussian_filter


class Preprocessor:
    """
    压力帧预处理器 — 支持底噪校准、死区滤波、中值/高斯空间滤波。

    Parameters
    ----------
    deadzone : int
        死区阈值，低于此值的像素置零。
    absolute_max : int
        硬件异常判定阈值，帧内最大值超过此值则该帧无效。
    sigma : float
        高斯滤波 sigma。设为 0 则跳过高斯滤波。
    """

    def __init__(self, deadzone=35, absolute_max=5000, sigma=0.8):
        self.deadzone = deadzone
        self.absolute_max = absolute_max
        self.sigma = sigma
        self.base_matrix = None

    def calibrate(self, calibration_frames):
        """使用前几帧计算底噪基准"""
        self.base_matrix = np.mean(calibration_frames, axis=0)
        print("预处理：底噪校准基准已建立")

    def is_frame_valid(self, frame):
        """检查帧是否包含硬件异常值(如21677)"""
        if np.max(frame) > self.absolute_max:
            return False
        return True

    def process_frame(self, frame):
        """单帧数据清洗: 减底噪 -> 死区 -> 中值滤波 -> 高斯平滑"""
        f = frame.astype(np.float32)

        # 1. 减去基准底噪
        if self.base_matrix is not None:
            f = np.maximum(0, f - self.base_matrix)

        # 2. 死区过滤
        f[f < self.deadzone] = 0

        # 3. 空间中值滤波
        f = median_filter(f, size=3)

        # 4. 二维高斯滤波 (sigma=0 则跳过)
        if self.sigma > 0:
            f = gaussian_filter(f, sigma=self.sigma)

        return f


def clean_dataset(timestamps, frames, calib_count=10, fs=10.0, trim_seconds=20,
                  use_gaussian=True, gaussian_sigma=0.8):
    """
    清洗整个数据集: 校准底噪 → 裁剪首尾 → 逐帧检验和清洗。

    合并了 Breath_Extraction 和 HeartbeatRate 的两种执行顺序:
      - Breath 版本: 先清洗后裁剪
      - Heartbeat 版本: 先裁剪后清洗
    统一采用 Heartbeat 的先裁剪后清洗策略 (更高效，减少无效计算)。

    Parameters
    ----------
    timestamps : list
        时间戳列表。
    frames : ndarray
        原始帧数组 (N, 32, 32)。
    calib_count : int
        用于底噪校准的前 N 帧 (必须在裁剪前)。
    fs : float
        采样率 (Hz)。
    trim_seconds : int
        首尾各裁剪的秒数。
    use_gaussian : bool
        是否启用高斯滤波。
    gaussian_sigma : float
        高斯滤波 sigma 值。

    Returns
    -------
    valid_times : list
    valid_frames : ndarray (N, 32, 32)
    """
    # 1. 底噪校准 (使用裁剪前的最初几帧)
    sigma = gaussian_sigma if use_gaussian else 0.0
    proc = Preprocessor(absolute_max=5000, sigma=sigma)
    proc.calibrate(frames[:calib_count])

    # 2. 裁剪首尾各 trim_seconds 秒
    trim_count = int(trim_seconds * fs)
    if len(frames) > 2 * trim_count:
        frames = frames[trim_count:-trim_count]
        if isinstance(timestamps, list):
            timestamps = timestamps[trim_count:-trim_count]
        else:
            timestamps = timestamps[trim_count:-trim_count]
        print(f"预处理：已自动切除首尾各 {trim_seconds}s ({trim_count} 帧)")
    else:
        print(f"警告：总帧数 ({len(frames)}) 不足 {trim_seconds * 2}s，跳过裁剪。")

    # 3. 逐帧检验和清洗
    valid_times = []
    valid_frames = []
    dropped_count = 0

    for i in range(len(frames)):
        if proc.is_frame_valid(frames[i]):
            clean_f = proc.process_frame(frames[i])
            valid_frames.append(clean_f)
            valid_times.append(timestamps[i])
        else:
            dropped_count += 1

    print(f"清洗完成: 剔除坏帧 {dropped_count} 帧, 剩余有效帧 {len(valid_frames)} 帧")
    return valid_times, np.array(valid_frames)
