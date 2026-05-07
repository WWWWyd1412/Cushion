import numpy as np
import os


def load_pressure_txt(file_path):
    """
    解析压力矩阵保存的 TXT 文件
    :param file_path: 文件路径
    :return: (timestamps, frames_array)
             timestamps: 列表，存储每帧的时间字符串
             frames_array: ndarray, 形状为 (N, 32, 32)，N 为总帧数
    """
    timestamps = []
    frames = []

    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return None, None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # 去除首尾空格并按空格分割
            parts = line.strip().split(' ')

            # 基础校验：时间戳(1) + 数据点(1024) = 1025 列
            if len(parts) < 1025:
                # 忽略可能存在的空行或损坏行
                continue

            try:
                # 1. 提取时间戳
                timestamps.append(parts[0])

                # 2. 提取 1024 个压力数据点并转换为 uint16[cite: 26]
                # 注意：如果数据中含有异常大值（如21684），uint16 可以兼容，但后期需预处理
                raw_data = np.array(parts[1:1025], dtype=np.uint16)

                # 3. 将一维数据还原为 32x32 矩阵
                matrix = raw_data.reshape((32, 32))
                frames.append(matrix)

            except ValueError as e:
                print(f"解析第 {line_num} 行时出错: {e}")
                continue

    # 将列表转换为三维 NumPy 数组 (Frame_Index, Row, Col)
    frames_array = np.array(frames)

    print(f"数据加载完成: 共读取 {len(frames_array)} 帧")
    return timestamps, frames_array


def get_session_info(timestamps, frames):
    """打印数据的基本统计信息"""
    if frames is None or len(frames) == 0:
        return

    duration_frames = len(frames)
    max_val = np.max(frames)
    min_val = np.min(frames)
    avg_val = np.mean(frames)

    print("-" * 30)
    print(f"采样总数: {duration_frames} 帧")
    print(f"起始时间: {timestamps[0]}")
    print(f"结束时间: {timestamps[-1]}")
    print(f"数值范围: [{min_val}, {max_val}]")
    print(f"平均亮度: {avg_val:.2f}")
    print("-" * 30)


# 模块自测逻辑
# if __name__ == "__main__":
#     # 你可以放一个测试文件路径在这里进行单脚本调试
#     test_path = "D:/1/bs/new_CUSHION/data/20260501_162541.txt"
#     t, f = load_pressure_txt(test_path)
#     get_session_info(t, f)