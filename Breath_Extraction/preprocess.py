import numpy as np
from scipy.ndimage import median_filter


class Preprocessor:
    def __init__(self, deadzone=35, absolute_max=5000):
        """
        :param deadzone: 死区阈值，低于此值设为0[cite: 17]
        :param absolute_max: 硬件异常判定阈值，超过此值则剔除整帧
        """
        self.deadzone = deadzone
        self.absolute_max = absolute_max
        self.base_matrix = None

    def calibrate(self, calibration_frames):
        """使用前几帧计算底噪基准"""
        self.base_matrix = np.mean(calibration_frames, axis=0)
        print("预处理：底噪校准基准已建立")

    def is_frame_valid(self, frame):
        """检查帧是否包含硬件异常值(如21677)"""
        # 如果帧内最大值超过阈值，判定为无效帧[cite: 27]
        if np.max(frame) > self.absolute_max:
            return False
        return True

    def process_frame(self, frame):
        """单帧数据清洗"""
        f = frame.astype(np.float32)

        # 1. 减去基准底噪[cite: 17]
        if self.base_matrix is not None:
            f = np.maximum(0, f - self.base_matrix)

        # 2. 死区过滤[cite: 17]
        f[f < self.deadzone] = 0

        # 3. 空间中值滤波剔除微小噪点[cite: 13]
        f = median_filter(f, size=3)
        return f


def clean_dataset(timestamps, frames, calib_count=10):
    """
    清洗整个数据集：剔除坏帧，处理好帧
    """
    proc = Preprocessor(absolute_max=5000)  # 设置5000为界限，剔除21677等[cite: 27]
    proc.calibrate(frames[:calib_count])

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

