#!/usr/bin/env python3
"""
Heartbeat Analysis CLI — 离线心跳算法批量对比
==============================================
无头模式: 加载数据 → 清洗 → 滑窗分析 (4种算法) → 输出HR统计表。
"""

import sys
import os
import time
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from cushion.core import load_pressure_txt, clean_dataset
from cushion.core.signal_utils import smooth_signal, calculate_bpm_peak, calculate_bpm_fpr
from cushion.algorithms.decomposition.emd import extract_emd
from cushion.algorithms.decomposition.vmd import extract_vmd
from cushion.algorithms.decomposition.acmd import extract_acmd
from cushion.algorithms.decomposition.vme import extract_vme
from cushion.heartbeat.config import HeartbeatConfig as CFG
from cushion.ui.sliding_window import generate_windows, compute_bpm_statistics


DATA_PATH = os.path.join(_project_root, "data", "20260501_162541.txt")
WINDOW_SEC = 15      # 秒
STEP_SEC = 3         # 秒
FS = 10.0


def main():
    print("=" * 70)
    print("  心跳 (BCG) 算法批量滑窗对比测试")
    print("=" * 70)

    print(f"\n[1] 加载数据: {DATA_PATH}")
    timestamps, frames = load_pressure_txt(DATA_PATH)
    if frames is None:
        print("错误: 数据加载失败!")
        return 1

    print("\n[2] 数据清洗...")
    clean_times, clean_frames = clean_dataset(
        timestamps, frames, calib_count=10, fs=FS,
        trim_seconds=20, use_gaussian=False)
    print(f"    有效帧数: {len(clean_frames)}")

    window_size = int(WINDOW_SEC * FS)
    step_size = int(STEP_SEC * FS)
    windows = generate_windows(len(clean_frames), window_size, step_size)
    print(f"\n[3] 滑窗: {len(windows)} 个窗口 ({WINDOW_SEC}s/窗口, {STEP_SEC}s/步长)")

    algorithms = {
        "EMD":  lambda f: extract_emd(f, FS, freq_band=CFG.FREQ_BAND, wavelet_alpha=CFG.WAVELET_ALPHA, use_vme_baseline=True, use_multicomponent=False),
        "VMD":  lambda f: extract_vmd(f, FS, K=CFG.VMD_K, alpha=CFG.VMD_ALPHA, freq_band=CFG.FREQ_BAND, wavelet_alpha=CFG.WAVELET_ALPHA, use_vme_baseline=True, use_multicomponent=False),
        "ACMD": lambda f: extract_acmd(f, FS),
        "VME":  lambda f: extract_vme(f, FS),
    }

    print(f"\n[4] 运行 {len(algorithms)} 种算法...\n")
    print(f"{'算法':<8} {'BPM方法':<10} {'均值':>7} {'标准差':>7} {'最小':>7} {'最大':>7} {'有效率':>7} {'耗时(s)':>8}")
    print("-" * 70)

    for algo_name, extract_fn in algorithms.items():
        t_start = time.time()
        for bpm_method, bpm_fn in [("FPR", calculate_bpm_fpr), ("Peak", calculate_bpm_peak)]:
            bpm_list = []
            for start, end in windows:
                win_frames = clean_frames[start:end]
                wave = extract_fn(win_frames)
                wave = smooth_signal(wave, window_size=CFG.SAVGOL_WINDOW,
                                     polyorder=CFG.SAVGOL_ORDER)
                if bpm_method == "FPR":
                    bpm = bpm_fn(wave, FS, min_dist_sec=CFG.BPM_MIN_DIST_SEC)
                else:
                    bpm = bpm_fn(wave, FS, min_dist_sec=CFG.BPM_MIN_DIST_SEC,
                                 prominence_ratio=CFG.BPM_PROMINENCE_RATIO)
                bpm_list.append(bpm)

            stats = compute_bpm_statistics(bpm_list)
            elapsed = time.time() - t_start
            print(f"{algo_name:<8} {bpm_method:<10} {stats['mean']:7.1f} {stats['std']:7.1f} "
                  f"{stats['min']:7.1f} {stats['max']:7.1f} {stats['valid_ratio']:6.1%} {elapsed:8.1f}")

    print("-" * 70)
    print("完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
