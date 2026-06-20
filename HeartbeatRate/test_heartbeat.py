import os
import sys
import numpy as np

# 加入当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import data_loader
import preprocess
import algorithms

def main():
    # 测试文件路径
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    test_file = os.path.join(data_dir, "20260501_162541.txt")
    
    if not os.path.exists(test_file):
        print(f"找不到测试文件: {test_file}")
        sys.exit(1)
        
    print(f"==== 1. 加载测试数据: {test_file} ====")
    timestamps, frames = data_loader.load_pressure_txt(test_file)
    if frames is None or len(frames) == 0:
        print("数据为空，退出测试")
        sys.exit(1)
        
    print(f"==== 2. 执行数据清洗 ====")
    clean_times, clean_frames = preprocess.clean_dataset(timestamps, frames)
    
    fs = 10.0
    print(f"数据采样率 fs = {fs} Hz, 有效帧数 = {len(clean_frames)}")

    # 对比四种提取方法
    algos = ["EMD", "VMD", "ACMD", "VME"]
    for algo in algos:
        print(f"\n==== 3. 运行算法: {algo} ====")
        try:
            if algo == "EMD":
                raw_hb = algorithms.extract_emd(clean_frames, fs)
            elif algo == "VMD":
                raw_hb = algorithms.extract_vmd(clean_frames, fs)
            elif algo == "ACMD":
                raw_hb = algorithms.extract_acmd(clean_frames, fs)
            elif algo == "VME":
                raw_hb = algorithms.extract_vme(clean_frames, fs)
                
            processed = raw_hb
            smoothed = algorithms.smooth_heartbeat_signal(processed)
            
            bpm_std = algorithms.calculate_bpm(smoothed, fs)
            bpm_fpr = algorithms.calculate_bpm_fpr(smoothed, fs)
            
            print(f"[{algo}] 方法")
            print(f"常规寻峰估测心率: {bpm_std:.2f} BPM")
            print(f"自适应 FPR 估测心率: {bpm_fpr:.2f} BPM")
            
        except Exception as e:
            print(f"[{algo}] 提取出错: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
