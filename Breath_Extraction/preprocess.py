import numpy as np
from scipy.ndimage import median_filter, gaussian_filter


class Preprocessor:
    def __init__(self, deadzone=35, absolute_max=5000, sigma=0.8):
        """
        :param deadzone: 死区阈值，低于此值设为0[cite: 17]
        :param absolute_max: 硬件异常判定阈值，超过此值则剔除整帧
        :param sigma: 高斯滤波方差
        """
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

        # 4. 二维高斯滤波平滑过度，协助稳定多 ROI 提取
        if self.sigma > 0:
            f = gaussian_filter(f, sigma=self.sigma)
            
        return f


def clean_dataset(timestamps, frames, calib_count=10):
    """
    清洗整个数据集：通过初始帧校准底噪，清洗数据，并去除前20s和后20s的有效数据
    """
    # 1. 首先在最开始的空载帧上完成底噪校准
    proc = Preprocessor(absolute_max=5000)  # 设置5000为界限，剔除21677等[cite: 27]
    proc.calibrate(frames[:calib_count])

    valid_times = []
    valid_frames = []
    dropped_count = 0

    # 2. 清洗整段数据
    for i in range(len(frames)):
        if proc.is_frame_valid(frames[i]):
            clean_f = proc.process_frame(frames[i])
            valid_frames.append(clean_f)
            valid_times.append(timestamps[i])
        else:
            dropped_count += 1

    print(f"清洗完成: 剔除坏帧 {dropped_count} 帧, 剩余有效帧 {len(valid_frames)} 帧")
    
    # 3. 采样率为 10Hz，20s 对应 200 帧。在清洗完成后，安全剪裁首尾各 20s (200 帧) 数据
    crop_frames = 200
    valid_frames_arr = np.array(valid_frames)
    
    if len(valid_frames_arr) > 2 * crop_frames:
        print(f"[Preprocessing] 正在去除首尾各20s数据 (各 {crop_frames} 帧)...")
        valid_frames_arr = valid_frames_arr[crop_frames:-crop_frames]
        valid_times = valid_times[crop_frames:-crop_frames]
    else:
        print(f"[Preprocessing] 警告：有效数据总长 ({len(valid_frames_arr)} 帧) 不足 40s，不进行裁剪。")
        
    return valid_times, valid_frames_arr



