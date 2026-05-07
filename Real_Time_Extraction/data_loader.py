import numpy as np
import os

def load_pressure_txt(file_path):
    """
    解析实时采集系统保存的 TXT 文件
    返回: (时间戳列表, (N, 32, 32) 形状的 NumPy 数组)
    """
    timestamps = []
    frames = []

    if not os.path.exists(file_path):
        print(f"文件未找到: {file_path}")
        return None, None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) < 1025: # 时间戳(1) + 1024个点
                continue
            try:
                timestamps.append(parts[0])
                # 转换 1024 个压力点为 uint16 并重排为 32x32
                raw_data = np.array(parts[1:1025], dtype=np.uint16)
                frames.append(raw_data.reshape((32, 32)))
            except ValueError:
                continue

    return timestamps, np.array(frames)