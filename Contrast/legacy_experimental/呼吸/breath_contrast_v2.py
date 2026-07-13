# -*- coding: utf-8 -*-
"""
多ROI呼吸算法对比分析脚本 v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
设计思路:
  1. 将40×40座垫分为左(cols0:20)、右(cols20:40)两区
  2. 各区按时间均值帧的压力大小，选取 K 个ROI中心
     (3×3窗口, 中心间距 ≥ MIN_DIST 像素, 避免重叠)
  3. 对每个ROI提取时序信号:
       去均值 → 小波去噪 → 带通0.1–0.5Hz
  4. 对每个ROI信号运行全套呼吸提取算法
  5. 滑动窗口(30s/步长5s)计算瞬时BPM曲线
  6. 对比参考RSP(CH1) —— 输出误差汇总+图表

预处理步骤(与现有模块一致):
  clip(0, 2000) → 死区(<30置0) → 中值滤波3×3 → 高斯平滑σ=0.5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys, os, csv, warnings
warnings.filterwarnings('ignore')

# ── 路径设置 ────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, '40_40_Extraction_1'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from datetime import datetime
from scipy.ndimage import median_filter, gaussian_filter
from scipy.signal import butter, filtfilt

# 导入算法
from algorithms.base import (
    calculate_bpm_fpr, butter_bandpass_filter, wavelet_denoise,
)
from algorithms.breath_extract import (
    extract_breath_mean, extract_breath_vmd,
    extract_breath_emd,  extract_breath_afd,
    extract_breath_vmd_mape, extract_breath_goa_vmd,
    extract_breath_smvmd, extract_breath_mvmd,
    extract_breath_multi_roi_ica, extract_breath_acmd,
)

# ── 全局配置 ────────────────────────────────────────────────────
FS          = 11.2          # 座垫固定采样率 (Hz)
TRIM_SEC    = 20.0          # 去头尾秒数
ROI_SIZE    = 3             # ROI窗口边长 (像素)
K_ROIS      = 4             # 每侧选取ROI数量
MIN_DIST    = 5             # ROI中心最小间距 (像素)
WIN_SEC     = 30.0          # 滑动窗口长度 (s)
STEP_SEC    = 5.0           # 滑动步长 (s)
BREATH_LOW  = 0.1           # 带通下限 (Hz)
BREATH_HIGH = 0.5           # 带通上限 (Hz)
DEADZONE    = 30            # 死区阈值
CLIP_MAX    = 2000          # 压力裁剪上限

DATA_FILE = os.path.join(PROJECT_ROOT, 'data',         '20260702_160410_40x40.txt')
REF_FILE  = os.path.join(PROJECT_ROOT, 'Precise_Data', '刘若红0702.txt')
OUT_DIR   = os.path.join(PROJECT_ROOT, 'Contrast', '呼吸', '刘若红_0702_160410_多ROI对比')

# 算法名称列表（保持顺序）
ALGO_NAMES = [
    '均值法', 'ACMD', 'VMD', 'EMD', 'AFD',
    'VMD-MAPE', 'GOA-VMD', 'SMVMD', 'MVMD', 'Multi-ROI ICA',
]


# ════════════════════════════════════════════════════════════════
# 工具: 中文字体
# ════════════════════════════════════════════════════════════════
def _setup_font():
    for name in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            plt.rcParams['font.family'] = name
            plt.rcParams['axes.unicode_minus'] = False
            return
        except Exception:
            pass
    plt.rcParams['axes.unicode_minus'] = False

_setup_font()


# ════════════════════════════════════════════════════════════════
# Part 1 — 预处理函数
# ════════════════════════════════════════════════════════════════
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    单帧预处理（与40_40_Extraction_1/preprocess.py保持一致）:
      1. float转换
      2. clip(0, CLIP_MAX)
      3. 死区阈值过滤
      4. 中值滤波 3×3
      5. 高斯平滑 sigma=0.5
    """
    f = frame.astype(np.float32)
    f = np.clip(f, 0, CLIP_MAX)
    f[f < DEADZONE] = 0.0
    f = median_filter(f, size=3)
    f = gaussian_filter(f, sigma=0.5)
    return f


# ════════════════════════════════════════════════════════════════
# Part 2 — 加载座垫数据
# ════════════════════════════════════════════════════════════════
def load_cushion_data(filepath: str):
    """
    读取40×40压力txt文件。
    格式: HH:MM:SS.ffffff  v1 v2 … v1600
    返回: frames (N,40,40) float32, timestamps (N,) float64
    """
    frames, timestamps = [], []
    print(f"[DATA] 加载座垫数据: {os.path.basename(filepath)}")

    with open(filepath, 'r', encoding='utf-8') as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 1601:
                continue
            try:
                t = datetime.strptime(parts[0], '%H:%M:%S.%f')
                ts = t.hour*3600 + t.minute*60 + t.second + t.microsecond/1e6
            except ValueError:
                continue
            raw = np.array(parts[1:1601], dtype=np.float32).reshape(40, 40)
            frames.append(preprocess_frame(raw))
            timestamps.append(ts)

    frames     = np.array(frames,     dtype=np.float32)
    timestamps = np.array(timestamps, dtype=np.float64)

    # 去头尾 TRIM_SEC 秒
    trim = int(TRIM_SEC * FS)
    if len(frames) > 2 * trim:
        frames     = frames[trim:-trim]
        timestamps = timestamps[trim:-trim]

    print(f"       → 有效帧 {len(frames)} 帧 ({len(frames)/FS:.1f}s) @ {FS} Hz")
    return frames, timestamps


# ════════════════════════════════════════════════════════════════
# Part 3 — 加载参考RSP信号
# ════════════════════════════════════════════════════════════════
def load_reference_rsp(filepath: str):
    """
    解析ACQ导出txt，提取CH1(RSP)。
    返回: rsp_ds (下采样至10Hz的带通信号), fs_ds, full_bpm
    """
    print(f"[REF]  加载参考RSP: {os.path.basename(filepath)}")
    fs_ref = 2000.0

    with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
        lines = fh.readlines()

    for ln in lines[:4]:
        if 'msec/sample' in ln:
            try:
                fs_ref = 1000.0 / float(ln.strip().split()[0])
            except ValueError:
                pass
            break

    data_start = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith('CH1'):
            data_start = i + 2
            break

    rsp_raw = []
    for ln in lines[data_start:]:
        cols = ln.strip().split('\t')
        try:
            rsp_raw.append(float(cols[0]))
        except (ValueError, IndexError):
            continue

    rsp_raw = np.array(rsp_raw, dtype=np.float64)
    print(f"       → {len(rsp_raw)} 点 @ {fs_ref:.0f} Hz ({len(rsp_raw)/fs_ref:.1f}s)")

    # 去头尾
    trim = int(TRIM_SEC * fs_ref)
    if len(rsp_raw) > 2 * trim:
        rsp_raw = rsp_raw[trim:-trim]

    rsp_raw -= rsp_raw.mean()

    # 低通抗混叠 → 下采样 → 带通
    nyq = 0.5 * fs_ref
    b, a = butter(4, 1.0 / nyq, btype='low')
    rsp_lp = filtfilt(b, a, rsp_raw)

    fs_ds = 10.0
    ds    = max(1, int(fs_ref / fs_ds))
    rsp_ds = rsp_lp[::ds]
    fs_ds_actual = fs_ref / ds

    rsp_bp = butter_bandpass_filter(rsp_ds, BREATH_LOW, BREATH_HIGH,
                                    fs=fs_ds_actual, order=4)

    full_bpm = calculate_bpm_fpr(rsp_bp, fs=fs_ds_actual, min_dist_s=1.5)
    print(f"       → 参考BPM = {full_bpm:.2f}")
    return rsp_bp, float(fs_ds_actual), float(full_bpm)


# ════════════════════════════════════════════════════════════════
# Part 4 — ROI中心选取（左/右分区，Top-K, 最小间距约束）
# ════════════════════════════════════════════════════════════════
def pick_roi_centers(mean_frame: np.ndarray, k: int, min_dist: int,
                     col_start: int, col_end: int):
    """
    在 mean_frame[:, col_start:col_end] 区域内，
    按压力值降序选取 k 个中心坐标（全局行列坐标）。
    相邻中心的切比雪夫距离必须 >= min_dist。
    """
    zone = mean_frame[:, col_start:col_end]
    # 平铺后按压力降序
    flat_order = np.argsort(zone.ravel())[::-1]
    centers = []

    for idx_flat in flat_order:
        r, c_local = np.unravel_index(idx_flat, zone.shape)
        c = c_local + col_start         # 转回全局列坐标

        # 检查与已选中心的切比雪夫距离
        too_close = any(
            max(abs(r - cr), abs(c - cc)) < min_dist
            for cr, cc in centers
        )
        if not too_close:
            centers.append((r, c))
        if len(centers) == k:
            break

    # 不足 k 个时用区域中心补充
    h, w = zone.shape
    while len(centers) < k:
        centers.append((h // 2, col_start + w // 2))

    return centers  # list of (row, col) in global coords


def select_all_rois(frames: np.ndarray):
    """
    对整段帧取时间均值帧，分别在左/右两区各选 K_ROIS 个ROI中心。
    返回:
        rois: list of dict {
            'label': 'L1'/'R3'等,
            'center': (r, c),
            'side':   'left'/'right'
        }
        mean_frame: (40,40) 均值帧（用于可视化）
    """
    mean_frame = frames.mean(axis=0)          # (40, 40)

    left_centers  = pick_roi_centers(mean_frame, K_ROIS, MIN_DIST, 0,  20)
    right_centers = pick_roi_centers(mean_frame, K_ROIS, MIN_DIST, 20, 40)

    rois = []
    for i, (r, c) in enumerate(left_centers):
        rois.append({'label': f'L{i+1}', 'center': (r, c), 'side': 'left'})
    for i, (r, c) in enumerate(right_centers):
        rois.append({'label': f'R{i+1}', 'center': (r, c), 'side': 'right'})

    print(f"[ROI]  选取 {len(rois)} 个ROI中心:")
    for roi in rois:
        r, c = roi['center']
        print(f"       {roi['label']}: 行{r:2d} 列{c:2d}  "
              f"均值压力={mean_frame[r, c]:.1f}")
    return rois, mean_frame


# ════════════════════════════════════════════════════════════════
# Part 5 — 从帧序列提取单个ROI的时序信号
# ════════════════════════════════════════════════════════════════
def extract_roi_signal(frames: np.ndarray, center: tuple) -> np.ndarray:
    """
    以 center=(r,c) 为中心，提取3×3窗口内的均值时序信号，
    并经过:
      1. 去均值（去直流）
      2. 小波去噪 (alpha=0.5)
      3. 带通滤波 0.1–0.5 Hz @ FS
    返回: 1D ndarray, 长度 = len(frames)
    """
    half = ROI_SIZE // 2
    r, c = center
    H, W = frames.shape[1], frames.shape[2]

    r_s = max(0,   r - half)
    r_e = min(H,   r + half + 1)
    c_s = max(0,   c - half)
    c_e = min(W,   c + half + 1)

    roi_ts = frames[:, r_s:r_e, c_s:c_e].mean(axis=(1, 2))  # (N,)

    # 去均值
    roi_ts = roi_ts - roi_ts.mean()
    # 小波去噪
    roi_ts = wavelet_denoise(roi_ts, alpha=0.5)
    # 带通
    roi_ts = butter_bandpass_filter(roi_ts, BREATH_LOW, BREATH_HIGH,
                                    fs=FS, order=3)
    return roi_ts.astype(np.float64)


# ════════════════════════════════════════════════════════════════
# Part 6 — 单段算法执行（给定1D信号 + frames备用）
# ════════════════════════════════════════════════════════════════
def run_algorithms_on_signal(signal: np.ndarray, frames: np.ndarray) -> dict:
    """
    对一段1D信号（已预处理）运行全套呼吸提取算法。
    frames 供需要3D输入的算法使用。
    返回: {algo_name: {'signal': ndarray, 'bpm': float}}
    """
    algo_funcs = {
        '均值法':        lambda: extract_breath_mean(signal),
        'ACMD':          lambda: extract_breath_acmd(signal,  fs=FS),
        'VMD':           lambda: extract_breath_vmd (signal,  fs=FS),
        'EMD':           lambda: extract_breath_emd (signal,  fs=FS),
        'AFD':           lambda: extract_breath_afd (signal,  fs=FS),
        'VMD-MAPE':      lambda: extract_breath_vmd_mape(frames,     fs=FS),
        'GOA-VMD':       lambda: extract_breath_goa_vmd(frames,      fs=FS),
        'SMVMD':         lambda: extract_breath_smvmd(frames,        fs=FS),
        'MVMD':          lambda: extract_breath_mvmd(frames,         fs=FS),
        'Multi-ROI ICA': lambda: extract_breath_multi_roi_ica(frames, fs=FS),
    }
    results = {}
    for name in ALGO_NAMES:
        try:
            sig_out = algo_funcs[name]()
            bpm     = calculate_bpm_fpr(sig_out, fs=FS, min_dist_s=1.5)
            results[name] = {'signal': sig_out, 'bpm': float(bpm)}
        except Exception as exc:
            results[name] = {'signal': np.zeros_like(signal), 'bpm': 0.0}
            print(f"         [warn] {name}: {exc}")
    return results


# ════════════════════════════════════════════════════════════════
# Part 7 — 滑动窗口BPM序列
# ════════════════════════════════════════════════════════════════
def sliding_bpm(signal: np.ndarray, fs: float,
                win_sec: float = WIN_SEC,
                step_sec: float = STEP_SEC) -> tuple:
    """
    在信号上以固定窗口/步长滑动，每窗口计算一个BPM。
    返回: (times_center, bpms)  均为1D ndarray
    """
    win  = int(win_sec  * fs)
    step = int(step_sec * fs)
    n    = len(signal)
    times, bpms = [], []

    i = 0
    while i + win <= n:
        seg = signal[i : i + win]
        bpm = calculate_bpm_fpr(seg, fs=fs, min_dist_s=1.5)
        times.append((i + win / 2) / fs)
        bpms.append(bpm)
        i += step

    return np.array(times), np.array(bpms)


def sliding_bpm_ref(ref_signal: np.ndarray, fs_ref: float) -> tuple:
    """参考RSP信号的滑动窗口BPM（采用相同窗口/步长参数）。"""
    return sliding_bpm(ref_signal, fs_ref)


# ════════════════════════════════════════════════════════════════
# Part 8 — 对所有ROI运行算法（全段 + 滑窗）
# ════════════════════════════════════════════════════════════════
def process_all_rois(rois: list, frames: np.ndarray):
    """
    对每个ROI:
      1. 提取预处理后的时序信号
      2. 运行所有算法（全段BPM）
      3. 对每种算法的输出做滑动窗口BPM

    返回 roi_results: list of {
        'label':   str,
        'side':    str,
        'center':  (r, c),
        'raw_sig': 1D ndarray,       # ROI预处理信号
        'algo':    {name: {'signal', 'bpm', 'sw_times', 'sw_bpms'}}
    }
    """
    roi_results = []
    total = len(rois)

    for idx, roi in enumerate(rois):
        label  = roi['label']
        center = roi['center']
        print(f"\n[ROI {idx+1}/{total}] {label} center=({center[0]},{center[1]})")

        # 提取并预处理ROI信号
        raw_sig = extract_roi_signal(frames, center)

        # 全段算法
        print(f"  全段算法:")
        algo_res = run_algorithms_on_signal(raw_sig, frames)
        for name in ALGO_NAMES:
            print(f"    {name:<16} BPM={algo_res[name]['bpm']:>6.2f}")

        # 滑动窗口BPM（对每种算法的输出信号再滑窗）
        for name in ALGO_NAMES:
            out_sig = algo_res[name]['signal']
            sw_t, sw_b = sliding_bpm(out_sig, fs=FS)
            algo_res[name]['sw_times'] = sw_t
            algo_res[name]['sw_bpms']  = sw_b

        # 同时对原始ROI信号本身也做滑窗（作为基准对比）
        raw_sw_t, raw_sw_b = sliding_bpm(raw_sig, fs=FS)

        roi_results.append({
            'label':      label,
            'side':       roi['side'],
            'center':     center,
            'raw_sig':    raw_sig,
            'raw_sw_t':   raw_sw_t,
            'raw_sw_b':   raw_sw_b,
            'algo':       algo_res,
        })

    return roi_results


# ════════════════════════════════════════════════════════════════
# Part 9 — 绘图：ROI位置可视化
# ════════════════════════════════════════════════════════════════
def plot_roi_positions(mean_frame: np.ndarray, rois: list, out_dir: str):
    """在均值压力热力图上标注所有ROI位置。"""
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(mean_frame, cmap='inferno', origin='upper')
    plt.colorbar(im, ax=ax, label='压力均值')

    # 左右分界线
    ax.axvline(x=19.5, color='cyan', lw=1.5, ls='--', label='左/右分界')

    colors = {'left': '#00e5ff', 'right': '#ff6e40'}
    for roi in rois:
        r, c = roi['center']
        side  = roi['side']
        label = roi['label']
        half  = ROI_SIZE // 2
        rect  = mpatches.Rectangle(
            (c - half - 0.5, r - half - 0.5),
            ROI_SIZE, ROI_SIZE,
            linewidth=2, edgecolor=colors[side], facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(c, r - half - 1.2, label,
                color=colors[side], fontsize=9, ha='center', va='bottom',
                fontweight='bold')

    ax.set_title('ROI选取位置 (左:青色 / 右:橙色)', fontsize=12)
    ax.set_xlabel('列'); ax.set_ylabel('行')
    ax.legend(loc='upper right', fontsize=9)

    out_path = os.path.join(out_dir, 'ROI位置可视化.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ROI可视化 → {out_path}")


# ════════════════════════════════════════════════════════════════
# Part 10 — 绘图：全段BPM误差热力图
#   行 = 算法, 列 = ROI,  色值 = 绝对误差(BPM)
# ════════════════════════════════════════════════════════════════
def plot_bpm_heatmap(roi_results: list, ref_bpm: float, out_dir: str):
    n_algo = len(ALGO_NAMES)
    n_roi  = len(roi_results)
    labels = [r['label'] for r in roi_results]

    error_mat = np.zeros((n_algo, n_roi))
    bpm_mat   = np.zeros((n_algo, n_roi))
    for j, rr in enumerate(roi_results):
        for i, name in enumerate(ALGO_NAMES):
            bpm_mat[i, j]   = rr['algo'][name]['bpm']
            error_mat[i, j] = abs(rr['algo'][name]['bpm'] - ref_bpm)

    fig, axes = plt.subplots(1, 2, figsize=(max(14, n_roi * 1.4 + 4), 7),
                              constrained_layout=True)
    fig.suptitle(f'全段BPM结果  (参考={ref_bpm:.2f} BPM)', fontsize=13)

    # 左图: 提取的BPM值
    im1 = axes[0].imshow(bpm_mat, cmap='RdYlGn', aspect='auto',
                         vmin=max(0, ref_bpm - 15), vmax=ref_bpm + 15)
    axes[0].set_xticks(range(n_roi));  axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_yticks(range(n_algo)); axes[0].set_yticklabels(ALGO_NAMES, fontsize=9)
    axes[0].set_title('各ROI × 算法  提取BPM值')
    plt.colorbar(im1, ax=axes[0], label='BPM')
    for i in range(n_algo):
        for j in range(n_roi):
            axes[0].text(j, i, f'{bpm_mat[i,j]:.1f}',
                         ha='center', va='center', fontsize=7,
                         color='black' if 0.3 < bpm_mat[i,j]/ref_bpm < 1.7 else 'red')

    # 右图: 绝对误差
    im2 = axes[1].imshow(error_mat, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=10)
    axes[1].set_xticks(range(n_roi));  axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_yticks(range(n_algo)); axes[1].set_yticklabels(ALGO_NAMES, fontsize=9)
    axes[1].set_title('绝对误差 |BPM - 参考|')
    plt.colorbar(im2, ax=axes[1], label='|误差| BPM')
    for i in range(n_algo):
        for j in range(n_roi):
            axes[1].text(j, i, f'{error_mat[i,j]:.1f}',
                         ha='center', va='center', fontsize=7)

    out_path = os.path.join(out_dir, 'BPM误差热力图.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  BPM热力图 → {out_path}")


# ════════════════════════════════════════════════════════════════
# Part 11 — 绘图：每种算法的滑动窗口BPM曲线
#   每张图: 所有ROI的该算法BPM曲线 + 参考RSP曲线
# ════════════════════════════════════════════════════════════════
def plot_sliding_bpm_per_algo(roi_results: list,
                               ref_sw_t: np.ndarray,
                               ref_sw_b: np.ndarray,
                               ref_bpm:  float,
                               out_dir:  str):
    """为每种算法单独绘制一张滑动窗口BPM对比图。"""
    palette_l = plt.cm.Blues  (np.linspace(0.4, 0.9, K_ROIS))
    palette_r = plt.cm.Oranges(np.linspace(0.4, 0.9, K_ROIS))

    left_rois  = [r for r in roi_results if r['side'] == 'left']
    right_rois = [r for r in roi_results if r['side'] == 'right']

    for algo_name in ALGO_NAMES:
        fig, ax = plt.subplots(figsize=(14, 5))

        # 参考线
        ax.plot(ref_sw_t, ref_sw_b, 'r-', lw=2.5,
                label=f'参考RSP  BPM={ref_bpm:.2f}', zorder=10)
        ax.axhline(ref_bpm, color='red', lw=1, ls=':', alpha=0.5)

        # 左侧ROI
        for k, rr in enumerate(left_rois):
            t = rr['algo'][algo_name]['sw_times']
            b = rr['algo'][algo_name]['sw_bpms']
            full_b = rr['algo'][algo_name]['bpm']
            ax.plot(t, b, color=palette_l[k], lw=1.5, marker='o',
                    markersize=3, label=f"{rr['label']} ({full_b:.1f})")

        # 右侧ROI
        for k, rr in enumerate(right_rois):
            t = rr['algo'][algo_name]['sw_times']
            b = rr['algo'][algo_name]['sw_bpms']
            full_b = rr['algo'][algo_name]['bpm']
            ax.plot(t, b, color=palette_r[k], lw=1.5, marker='s',
                    markersize=3, label=f"{rr['label']} ({full_b:.1f})",
                    linestyle='--')

        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('BPM')
        ax.set_title(f'[{algo_name}] 滑动窗口BPM对比  '
                     f'(窗口{WIN_SEC:.0f}s / 步长{STEP_SEC:.0f}s)')
        ax.legend(fontsize=8, ncol=3, loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, max(40, ref_bpm * 2.5))

        safe_name = algo_name.replace(' ', '_').replace('/', '-')
        out_path  = os.path.join(out_dir, f'滑窗BPM_{safe_name}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  滑窗BPM图 → {out_dir}/滑窗BPM_*.png")


# ════════════════════════════════════════════════════════════════
# Part 12 — CSV导出
# ════════════════════════════════════════════════════════════════
def save_csv(roi_results: list, ref_bpm: float, out_dir: str):
    """
    输出两个CSV:
    1. 全段BPM汇总表  (ROI × 算法)
    2. 每ROI每算法全段误差排名
    """
    # ── 全段BPM汇总 ──
    path1 = os.path.join(out_dir, '全段BPM汇总.csv')
    with open(path1, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        header = ['算法', '参考BPM'] + [r['label'] for r in roi_results]
        w.writerow(header)
        for name in ALGO_NAMES:
            row = [name, f'{ref_bpm:.3f}']
            for rr in roi_results:
                row.append(f"{rr['algo'][name]['bpm']:.3f}")
            w.writerow(row)
    print(f"  全段BPM汇总 → {path1}")

    # ── 误差汇总 ──
    path2 = os.path.join(out_dir, '误差汇总.csv')
    with open(path2, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['ROI', '侧', '算法', 'BPM', '参考BPM',
                    '绝对误差', '相对误差(%)'])
        for rr in roi_results:
            for name in ALGO_NAMES:
                b   = rr['algo'][name]['bpm']
                err = abs(b - ref_bpm)
                rel = err / ref_bpm * 100 if ref_bpm > 0 else float('nan')
                w.writerow([rr['label'], rr['side'], name,
                             f'{b:.3f}', f'{ref_bpm:.3f}',
                             f'{err:.3f}', f'{rel:.2f}'])
    print(f"  误差汇总   → {path2}")


# ════════════════════════════════════════════════════════════════
# Part 13 — 主流程
# ════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print('\n' + '='*60)
    print('  多ROI呼吸算法对比分析 v2.0')
    print(f'  输出目录: {OUT_DIR}')
    print('='*60)

    # ── 1. 加载并预处理座垫数据 ──
    frames, timestamps = load_cushion_data(DATA_FILE)

    # ── 2. 加载参考RSP ──
    ref_ds, ref_fs_ds, ref_bpm = load_reference_rsp(REF_FILE)

    # ── 3. ROI选取 ──
    rois, mean_frame = select_all_rois(frames)

    # ── 4. 可视化ROI位置 ──
    print('\n[PLOT] 生成图表...')
    plot_roi_positions(mean_frame, rois, OUT_DIR)

    # ── 5. 对所有ROI运行算法（含滑窗） ──
    print('\n[ALGO] 开始算法处理...')
    roi_results = process_all_rois(rois, frames)

    # ── 6. 参考RSP滑动窗口 ──
    ref_sw_t, ref_sw_b = sliding_bpm_ref(ref_ds, ref_fs_ds)

    # ── 7. 绘图输出 ──
    print('\n[PLOT] 绘制结果图...')
    plot_bpm_heatmap(roi_results, ref_bpm, OUT_DIR)
    plot_sliding_bpm_per_algo(roi_results, ref_sw_t, ref_sw_b, ref_bpm, OUT_DIR)

    # ── 8. CSV ──
    print('\n[CSV]  保存报告...')
    save_csv(roi_results, ref_bpm, OUT_DIR)

    # ── 9. 控制台汇总（每算法的最优ROI） ──
    print('\n' + '='*60)
    print(f'  参考BPM (RSP CH1) = {ref_bpm:.2f}')
    print(f'  {"算法":<16}  {"最优ROI":>6}  {"BPM":>7}  {"误差":>7}')
    print(f'  {"-"*44}')
    for name in ALGO_NAMES:
        best_roi = min(roi_results,
                       key=lambda r: abs(r['algo'][name]['bpm'] - ref_bpm))
        b   = best_roi['algo'][name]['bpm']
        err = abs(b - ref_bpm)
        flag = 'OK' if err <= 1.5 else ('~' if err <= 3.0 else 'X')
        print(f'  {name:<16}  {best_roi["label"]:>6}  {b:>7.2f}  '
              f'{err:>7.2f}  {flag}')
    print('='*60 + '\n')
    print(f'完成！结果已保存至: {OUT_DIR}\n')


if __name__ == '__main__':
    main()


