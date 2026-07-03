# -*- coding: utf-8 -*-
"""
呼吸算法对比分析脚本 v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入:
  - 座垫压力数据 : data/20260702_160410_40x40.txt
  - 精确参考信号 : Precise_Data/刘若红0702.txt  (CH1 = RSP)
输出 (自动创建):
  - Contrast/刘若红_0702_160410_对比结果/
      ├── 呼吸信号波形对比.png
      ├── BPM对比柱状图.png
      └── 结果报告.csv
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import os
import csv
import warnings
warnings.filterwarnings('ignore')

# ── 路径配置 ────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))  # 文件在 Contrast/呼吸/ 下，上溯两层
sys.path.insert(0, os.path.join(PROJECT_ROOT, '40_40_Extraction_1'))

import numpy as np
import matplotlib
matplotlib.use('Agg')          # 无 GUI 后端，避免弹窗
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime

from algorithms.base import (
    get_spatial_sum, smooth_signal,
    calculate_bpm_fpr, butter_bandpass_filter, wavelet_denoise,
)
from algorithms.breath_extract import (
    extract_breath_mean, extract_breath_vmd,
    extract_breath_emd,  extract_breath_afd,
    extract_breath_vmd_mape, extract_breath_goa_vmd,
    extract_breath_smvmd, extract_breath_mvmd,
    extract_breath_multi_roi_ica, extract_breath_acmd,
)
from preprocess import Preprocessor

# ── 文件路径 ────────────────────────────────────────────────────
DATA_FILE = os.path.join(PROJECT_ROOT, 'data',         '20260702_160410_40x40.txt')
REF_FILE  = os.path.join(PROJECT_ROOT, 'Precise_Data', '刘若红0702.txt')
OUT_DIR   = os.path.join(PROJECT_ROOT, 'Contrast', '呼吸', '刘若红_0702_160410_对比结果')


# ════════════════════════════════════════════════════════════════
# 1. 中文字体
# ════════════════════════════════════════════════════════════════
def _setup_font():
    for name in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            plt.rcParams['font.family'] = name
            plt.rcParams['axes.unicode_minus'] = False
            return
        except Exception:
            pass
    # 最后回退：用 matplotlib 默认字体，中文可能显示为方块，但不崩溃
    plt.rcParams['axes.unicode_minus'] = False

_setup_font()


# ════════════════════════════════════════════════════════════════
# 2. 加载座垫数据
# ════════════════════════════════════════════════════════════════
def load_cushion_data(filepath: str):
    """
    读取40×40压力文本文件。
    每行格式：HH:MM:SS.ffffff  v1 v2 … v1600
    返回:
        frames     (N, 40, 40) float32
        timestamps (N,)        float64  秒
    """
    preprocessor = Preprocessor(deadzone=30)
    frames, timestamps = [], []

    print(f"[1] 加载座垫数据: {os.path.basename(filepath)}")
    with open(filepath, 'r', encoding='utf-8') as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 1601:
                continue
            try:
                t = datetime.strptime(parts[0], '%H:%M:%S.%f')
                ts = t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
            except ValueError:
                continue
            vals = np.array(parts[1:1601], dtype=np.float32).reshape(40, 40)
            frames.append(preprocessor.process_frame(vals))
            timestamps.append(ts)

    frames     = np.array(frames,     dtype=np.float32)
    timestamps = np.array(timestamps, dtype=np.float64)
    print(f"    → {len(frames)} 帧")
    return frames, timestamps


def estimate_fs(timestamps: np.ndarray) -> float:
    """从时间戳差值估算采样率 (Hz)。"""
    diffs = np.diff(timestamps)
    diffs = diffs[(diffs > 0) & (diffs < 1.0)]   # 去掉跳变异常值
    return float(1.0 / np.mean(diffs)) if len(diffs) > 0 else 10.0


# ════════════════════════════════════════════════════════════════
# 3. 加载参考 RSP 信号
# ════════════════════════════════════════════════════════════════
def load_reference_rsp(filepath: str):
    """
    解析 AcqKnowledge ACQ 导出的 txt 文件，提取 CH1 (RSP) 列。
    文件头结构:
        行1  : 文件名
        行2  : "0.5 msec/sample"
        行3  : "3 channels"
        行4-6: 通道名称
        行7-9: 单位
        行10 : 列标题 "CH1\\tCH2\\tCH13\\t"
        行11 : 各通道采样点数
        行12+: 数值数据
    返回:
        rsp (N,) float64, fs_ref float
    """
    print(f"[2] 加载参考RSP信号: {os.path.basename(filepath)}")
    fs_ref = 2000.0

    with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
        lines = fh.readlines()

    # 解析采样率
    for ln in lines[:4]:
        if 'msec/sample' in ln:
            try:
                ms_per_sample = float(ln.strip().split()[0])
                fs_ref = 1000.0 / ms_per_sample
            except ValueError:
                pass
            break

    # 定位数据起始行（跳过列标题行 + 样本数行）
    data_start = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith('CH1'):
            data_start = i + 2      # CH1 标题行 + 样本数行
            break

    rsp = []
    for ln in lines[data_start:]:
        cols = ln.strip().split('\t')
        try:
            rsp.append(float(cols[0]))
        except (ValueError, IndexError):
            continue

    rsp = np.array(rsp, dtype=np.float64)
    print(f"    → {len(rsp)} 点 @ {fs_ref:.0f} Hz ({len(rsp)/fs_ref:.1f} s)")
    return rsp, fs_ref


# ════════════════════════════════════════════════════════════════
# 4. 从参考信号计算 BPM
# ════════════════════════════════════════════════════════════════
def ref_bpm_and_signal(rsp: np.ndarray, fs_ref: float, trim_sec: float = 20.0):
    """
    去头尾20s → 低通抗混叠 → 下采样至10Hz → 带通0.1-0.5Hz → FPR-BPM。

    注意：直接在2000Hz上做0.1-0.5Hz带通会因归一化频率极低(0.0001)
    导致 butter() 数值不稳定，因此先低通降采样再滤波。
    """
    from scipy.signal import butter, filtfilt

    trim = int(trim_sec * fs_ref)
    if len(rsp) > 2 * trim:
        rsp = rsp[trim:-trim]
    rsp = rsp - rsp.mean()

    # 1) 低通抗混叠：截止1Hz（充分覆盖呼吸频段）
    nyq  = 0.5 * fs_ref
    cut  = 1.0 / nyq   # 归一化截止频率
    if cut < 1.0:
        b, a = butter(4, cut, btype='low')
        rsp = filtfilt(b, a, rsp)

    # 2) 下采样到目标率
    fs_ds = 10.0
    ds    = max(1, int(fs_ref / fs_ds))
    rsp_ds = rsp[::ds]
    fs_actual = fs_ref / ds   # 实际下采样后频率

    # 3) 带通0.1-0.5Hz（在10Hz下计算稳定）
    rsp_bp = butter_bandpass_filter(rsp_ds, lowcut=0.1, highcut=0.5,
                                    fs=fs_actual, order=4)

    bpm = calculate_bpm_fpr(rsp_bp, fs=fs_actual, min_dist_s=1.5)
    return float(bpm), rsp_bp, float(fs_actual)


# ════════════════════════════════════════════════════════════════
# 5. 运行所有呼吸提取算法
# ════════════════════════════════════════════════════════════════
def run_all_algorithms(frames: np.ndarray, fs: float):
    """
    对 (N,40,40) 帧运行全套呼吸算法，返回结果字典。
    result[name] = {'signal': 1D ndarray, 'bpm': float}
    """
    # 1D 基础信号（去均值）
    sig1d = get_spatial_sum(frames, pressure_threshold=100)

    algo_map = {
        '均值法':         lambda: extract_breath_mean(sig1d),
        'ACMD':           lambda: extract_breath_acmd(sig1d,  fs=fs),
        'VMD':            lambda: extract_breath_vmd (sig1d,  fs=fs),
        'EMD':            lambda: extract_breath_emd (sig1d,  fs=fs),
        'AFD':            lambda: extract_breath_afd (sig1d,  fs=fs),
        'VMD-MAPE':       lambda: extract_breath_vmd_mape(frames,  fs=fs),
        'GOA-VMD':        lambda: extract_breath_goa_vmd(frames,   fs=fs),
        'SMVMD':          lambda: extract_breath_smvmd(frames,     fs=fs),
        'MVMD':           lambda: extract_breath_mvmd(frames,      fs=fs),
        'Multi-ROI ICA':  lambda: extract_breath_multi_roi_ica(frames, fs=fs),
    }

    results = {}
    print("\n[3] 运行呼吸提取算法:")
    for name, fn in algo_map.items():
        print(f"    {name:<16} ... ", end='', flush=True)
        try:
            sig = fn()
            bpm = calculate_bpm_fpr(sig, fs=fs, min_dist_s=1.5)
            results[name] = {'signal': sig, 'bpm': float(bpm)}
            print(f"BPM = {bpm:.2f}")
        except Exception as exc:
            print(f"[异常] {exc}")
            results[name] = {'signal': np.zeros(len(sig1d)), 'bpm': 0.0}

    return results, sig1d


# ════════════════════════════════════════════════════════════════
# 6. 绘图：波形对比
# ════════════════════════════════════════════════════════════════
def plot_waveforms(results, sig1d, ref_bpm, ref_ds, ref_fs_ds,
                   fs_cushion, out_dir):
    n = len(results)
    t_ref  = np.arange(len(ref_ds)) / ref_fs_ds
    t_cush = np.arange(len(sig1d)) / fs_cushion

    fig, axes = plt.subplots(n + 1, 1,
                              figsize=(16, 3.2 * (n + 1)),
                              constrained_layout=True)
    fig.suptitle('呼吸信号提取算法对比 — 刘若红 0702 (16:04:10)',
                 fontsize=14, fontweight='bold')

    # ── 参考信号 ──
    ax0 = axes[0]
    ax0.plot(t_ref, ref_ds, color='#27ae60', linewidth=1.3)
    ax0.set_title(f'参考 RSP (CH1) — BPM = {ref_bpm:.2f}', fontsize=11)
    ax0.set_ylabel('幅值 (V)')
    ax0.axhline(0, color='gray', lw=0.6, ls='--')
    ax0.grid(alpha=0.25)

    # ── 各算法 ──
    palette = plt.cm.tab10(np.linspace(0, 0.9, n))
    for idx, (name, res) in enumerate(results.items()):
        ax  = axes[idx + 1]
        sig = res['signal']
        t   = np.arange(len(sig)) / fs_cushion
        err = abs(res['bpm'] - ref_bpm)
        color_str = '#27ae60' if err <= 1.5 else ('#f39c12' if err <= 3.0 else '#e74c3c')

        ax.plot(t, sig, color=palette[idx], linewidth=1.2)
        ax.set_title(
            f'{name}  —  BPM = {res["bpm"]:.2f}  '
            f'(误差 {err:.2f} BPM)',
            fontsize=10, color=color_str
        )
        ax.set_ylabel('幅值')
        ax.axhline(0, color='gray', lw=0.6, ls='--')
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel('时间 (s)')

    out_path = os.path.join(out_dir, '呼吸信号波形对比.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    波形图 → {out_path}")


# ════════════════════════════════════════════════════════════════
# 7. 绘图：BPM 柱状图
# ════════════════════════════════════════════════════════════════
def plot_bpm_bar(results, ref_bpm, out_dir):
    names  = list(results.keys())
    bpms   = [results[n]['bpm']          for n in names]
    errors = [abs(results[n]['bpm'] - ref_bpm) for n in names]

    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                    constrained_layout=True)
    fig.suptitle('各算法呼吸BPM对比 — 刘若红 0702',
                 fontsize=13, fontweight='bold')

    # ── BPM ──
    bars1 = ax1.bar(x, bpms, width=0.62, color='#3498db', alpha=0.82, zorder=3)
    ax1.axhline(ref_bpm, color='#e74c3c', lw=2.2, ls='--',
                label=f'参考BPM = {ref_bpm:.2f}')
    ax1.set_xticks(x);  ax1.set_xticklabels(names, rotation=30, ha='right')
    ax1.set_ylabel('呼吸率 (BPM)')
    ax1.set_title('算法提取BPM vs 参考BPM')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.25, axis='y', zorder=0)
    for bar, v in zip(bars1, bpms):
        if v > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.15,
                     f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    # ── 误差 ──
    err_colors = ['#27ae60' if e <= 1.5 else '#f39c12' if e <= 3.0 else '#e74c3c'
                  for e in errors]
    bars2 = ax2.bar(x, errors, width=0.62, color=err_colors, alpha=0.85, zorder=3)
    ax2.axhline(1.5, color='#27ae60', lw=1.5, ls=':', label='±1.5 BPM 阈值')
    ax2.axhline(3.0, color='#f39c12', lw=1.5, ls=':', label='±3.0 BPM 阈值')
    ax2.set_xticks(x);  ax2.set_xticklabels(names, rotation=30, ha='right')
    ax2.set_ylabel('|误差| (BPM)')
    ax2.set_title('BPM绝对误差  (绿 ≤1.5 / 橙 ≤3.0 / 红 >3.0)')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.25, axis='y', zorder=0)
    for bar, e in zip(bars2, errors):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05,
                 f'{e:.2f}', ha='center', va='bottom', fontsize=9)

    out_path = os.path.join(out_dir, 'BPM对比柱状图.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    BPM柱状图 → {out_path}")


# ════════════════════════════════════════════════════════════════
# 8. 保存 CSV 报告
# ════════════════════════════════════════════════════════════════
def save_csv(results, ref_bpm, out_dir):
    out_path = os.path.join(out_dir, '结果报告.csv')
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['算法', '提取BPM', '参考BPM', '绝对误差(BPM)', '相对误差(%)'])
        for name, res in results.items():
            b   = res['bpm']
            err = abs(b - ref_bpm)
            rel = err / ref_bpm * 100 if ref_bpm > 0 else float('nan')
            w.writerow([name,
                        f'{b:.3f}', f'{ref_bpm:.3f}',
                        f'{err:.3f}', f'{rel:.2f}'])
    print(f"    CSV报告 → {out_path}")


# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('\n' + '═' * 60)
    print('  呼吸算法对比分析')
    print(f'  输出目录: {OUT_DIR}')
    print('═' * 60)

    # ── 1. 座垫数据 ──
    frames, timestamps = load_cushion_data(DATA_FILE)
    fs = estimate_fs(timestamps)
    print(f"    采样率 ≈ {fs:.2f} Hz，共 {len(frames)} 帧 "
          f"({len(frames)/fs:.1f} s)")

    # 去头尾20s（与提取模块保持一致）
    trim = int(20 * fs)
    if len(frames) > 2 * trim:
        frames = frames[trim:-trim]
        print(f"    裁剪后: {len(frames)} 帧 ({len(frames)/fs:.1f} s)")

    # ── 2. 参考信号 ──
    rsp_raw, fs_ref = load_reference_rsp(REF_FILE)
    ref_bpm, ref_ds, ref_fs_ds = ref_bpm_and_signal(rsp_raw, fs_ref,
                                                     trim_sec=20.0)
    print(f"    参考呼吸率 (RSP CH1): {ref_bpm:.2f} BPM")

    # ── 3. 算法对比 ──
    results, sig1d = run_all_algorithms(frames, fs)

    # ── 4. 输出 ──
    print('\n[4] 生成输出文件:')
    plot_waveforms(results, sig1d, ref_bpm, ref_ds, ref_fs_ds, fs, OUT_DIR)
    plot_bpm_bar  (results, ref_bpm, OUT_DIR)
    save_csv      (results, ref_bpm, OUT_DIR)

    # ── 5. 控制台汇总 ──
    print('\n' + '═' * 60)
    print(f"  {'算法':<18} {'BPM':>8}  {'误差':>8}  {'相对误差':>10}")
    print(f"  {'-' * 46}")
    for name, res in results.items():
        b   = res['bpm']
        err = abs(b - ref_bpm)
        rel = err / ref_bpm * 100 if ref_bpm > 0 else float('nan')
        flag = 'OK' if err <= 1.5 else ('~' if err <= 3.0 else 'X')
        print(f"  {name:<18} {b:>8.2f}  {err:>8.2f}  {rel:>9.1f}%  {flag}")
    print(f"  {'参考 RSP':<18} {ref_bpm:>8.2f}")
    print('═' * 60 + '\n')
    print(f'完成！结果已保存至: {OUT_DIR}\n')


if __name__ == '__main__':
    main()
