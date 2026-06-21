#!/usr/bin/env python3
"""
Breath Analysis CLI — 离线呼吸算法批量对比
===========================================
无头模式: 加载数据 → 清洗 → 滑窗分析 (7种算法) → 输出BPM统计表。
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
from cushion.algorithms.decomposition.smvmd import extract_smvmd
from cushion.algorithms.decomposition.mvmd import extract_mvmd
from cushion.algorithms.fusion.ica import extract_multi_roi_ica
from cushion.breath.config import BreathConfig as CFG
from cushion.ui.sliding_window import generate_windows, compute_bpm_statistics


DATA_PATH = os.path.join(_project_root, "data", "20260501_162541.txt")
WINDOW_SIZE = 250   # 帧
STEP_SIZE = 50      # 帧
FS = 10.0


def main():
    print("=" * 70)
    print("  呼吸算法批量滑窗对比测试")
    print("=" * 70)

    # 加载数据
    print(f"\n[1] 加载数据: {DATA_PATH}")
    timestamps, frames = load_pressure_txt(DATA_PATH)
    if frames is None:
        print("错误: 数据加载失败!")
        return 1

    # 清洗
    print("\n[2] 数据清洗...")
    clean_times, clean_frames = clean_dataset(
        timestamps, frames, calib_count=10, fs=FS,
        trim_seconds=20, use_gaussian=True, gaussian_sigma=CFG.GAUSSIAN_SIGMA)
    print(f"    有效帧数: {len(clean_frames)}")

    # 生成窗口
    windows = generate_windows(len(clean_frames), WINDOW_SIZE, STEP_SIZE)
    print(f"\n[3] 滑窗: {len(windows)} 个窗口 (窗口={WINDOW_SIZE}帧, 步长={STEP_SIZE}帧)")

    # 算法列表
    algorithms = {
        "EMD":             lambda f: extract_emd(f, FS, freq_band=CFG.FREQ_BAND, wavelet_alpha=CFG.WAVELET_ALPHA),
        "VMD":             lambda f: extract_vmd(f, FS, K=CFG.VMD_K, alpha=CFG.VMD_ALPHA, freq_band=CFG.FREQ_BAND, wavelet_alpha=CFG.WAVELET_ALPHA),
        "AFD":             lambda f: _extract_afd(f, FS),
        "VMD_FPR(MAPE)":   lambda f: _extract_vmd_mape(f, FS),
        "SMVMD":           lambda f: extract_smvmd(f, FS),
        "MVMD":            lambda f: extract_mvmd(f, FS),
        "Multi-ROI ICA":   lambda f: extract_multi_roi_ica(f, FS),
    }

    # 逐算法跑
    print(f"\n[4] 运行 {len(algorithms)} 种算法...\n")
    print(f"{'算法':<18} {'BPM方法':<10} {'均值':>7} {'标准差':>7} {'最小':>7} {'最大':>7} {'有效率':>7} {'耗时(s)':>8}")
    print("-" * 80)

    for algo_name, extract_fn in algorithms.items():
        t_start = time.time()
        for bpm_method, bpm_fn in [("Peak", calculate_bpm_peak), ("FPR", calculate_bpm_fpr)]:
            bpm_list = []
            for start, end in windows:
                win_frames = clean_frames[start:end]
                wave = extract_fn(win_frames)
                wave = smooth_signal(wave, window_size=CFG.SAVGOL_WINDOW,
                                     polyorder=CFG.SAVGOL_ORDER)
                if bpm_method == "Peak":
                    bpm = bpm_fn(wave, FS, min_dist_sec=CFG.BPM_MIN_DIST_SEC)
                else:
                    bpm = bpm_fn(wave, FS, min_dist_sec=CFG.BPM_MIN_DIST_SEC)
                bpm_list.append(bpm)

            stats = compute_bpm_statistics(bpm_list)
            elapsed = time.time() - t_start
            print(f"{algo_name:<18} {bpm_method:<10} {stats['mean']:7.1f} {stats['std']:7.1f} "
                  f"{stats['min']:7.1f} {stats['max']:7.1f} {stats['valid_ratio']:6.1%} {elapsed:8.1f}")

    print("-" * 80)
    print("完成!")
    return 0


def _extract_afd(frames, fs):
    from scipy.signal import hilbert
    from cushion.algorithms.base import get_dual_roi_mean, reconstruct_multicomponent_with_snr
    signal_1d = get_dual_roi_mean(frames, fs=fs, freq_band=CFG.FREQ_BAND, wavelet_alpha=0.5)
    z = hilbert(signal_1d - np.mean(signal_1d))
    t = np.arange(len(z)) / fs
    residual = z.copy()
    components = []
    for _ in range(5):
        best_comp = None
        max_proj = -1
        for f0 in np.linspace(0.1, 0.5, 50):
            kernel = np.exp(1j * 2 * np.pi * f0 * t)
            proj = np.abs(np.vdot(residual, kernel))
            if proj > max_proj:
                max_proj = proj
                best_comp = (np.vdot(residual, kernel) / np.vdot(kernel, kernel)) * kernel
        if best_comp is not None:
            components.append(np.real(best_comp))
            residual -= best_comp
    return reconstruct_multicomponent_with_snr(np.array(components), fs, freq_band=CFG.FREQ_BAND)


def _extract_vmd_mape(frames, fs):
    from vmdpy import VMD
    from cushion.algorithms.base import get_dual_roi_mean, reconstruct_multicomponent_with_snr
    signal_1d = get_dual_roi_mean(frames, fs=fs, freq_band=CFG.FREQ_BAND, wavelet_alpha=0.5)
    if len(signal_1d) < 100:
        return signal_1d
    mapes = []
    best_u = None
    for k in range(2, 11):
        u, _, _ = VMD(signal_1d, alpha=2000, tau=0, K=k, DC=0, init=1, tol=1e-7)
        res = signal_1d - np.sum(u, axis=0)
        mape = np.sum(res ** 2) / np.sum(signal_1d ** 2)
        if len(mapes) > 0 and mape > mapes[-1]:
            break
        mapes.append(mape)
        best_u = u
    return reconstruct_multicomponent_with_snr(best_u, fs, freq_band=CFG.FREQ_BAND)


if __name__ == "__main__":
    sys.exit(main())
