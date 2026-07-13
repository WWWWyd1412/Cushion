# -*- coding: utf-8 -*-
"""
呼吸算法改进版 v3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
针对 v2.0 中表现最优的三种算法 (ACMD / GOA-VMD / VMD) 进行改进:

改进1: 多ROI ICA融合信号作为算法输入
  - 收集所有8个(3×3)ROI的时序信号
  - ICA分离取呼吸频段主分量，作为更干净的1D输入

改进2: 自相关BPM估计 (AutoCorr-BPM)
  - 代替FPR峰值计数，直接找信号基础周期
  - 避免"吸气+呼气"双峰被当成两次呼吸的频率倍增问题

改进3: GOA-VMD帧范围限制
  - 不再传全帧，只传最优左/右ROI邻域子帧
  - 减少无关区域压力噪声的干扰

改进4: ACMD/VMD 在融合信号上运行 vs 单ROI信号
  - 对比两种输入方式的BPM精度
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys, os, csv, warnings
warnings.filterwarnings('ignore')

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, '40_40_Extraction_1'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime
from scipy.ndimage import median_filter, gaussian_filter
from scipy.signal import butter, filtfilt, find_peaks, correlate

from algorithms.base import (
    calculate_bpm_fpr, butter_bandpass_filter, wavelet_denoise,
)
from algorithms.breath_extract import (
    extract_breath_acmd, extract_breath_vmd, extract_breath_goa_vmd,
    extract_breath_mean,
)

# ── 配置 (与v2保持一致) ─────────────────────────────────────────
FS         = 11.2
TRIM_SEC   = 20.0
ROI_SIZE   = 3
K_ROIS     = 4
MIN_DIST   = 5
WIN_SEC    = 30.0
STEP_SEC   = 5.0
DEADZONE   = 30
CLIP_MAX   = 2000

DATA_FILE = os.path.join(PROJECT_ROOT, 'data',         '20260702_160410_40x40.txt')
REF_FILE  = os.path.join(PROJECT_ROOT, 'Precise_Data', '刘若红0702.txt')
OUT_DIR   = os.path.join(PROJECT_ROOT, 'Contrast', '呼吸', '刘若红_0702_160410_改进版')


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
# 改进1: 自相关BPM估计
# ════════════════════════════════════════════════════════════════
def calculate_bpm_autocorr(signal: np.ndarray, fs: float,
                            min_bpm: float = 6.0,
                            max_bpm: float = 40.0) -> float:
    """
    基于无偏归一化自相关的BPM估计。
    算法:
      1. 信号去均值
      2. 计算全长自相关，取正延迟部分
      3. 在 [min_period, max_period] 范围内找第一个主峰
      4. BPM = 60 / lag_seconds
    优势: 直接找基础周期，不受吸/呼双峰导致的2倍频影响。
    """
    n = len(signal)
    if n < 20:
        return 0.0

    s = signal - signal.mean()
    # 无偏归一化自相关
    acf = correlate(s, s, mode='full')
    acf = acf[n - 1:]                    # 取 lag >= 0 部分
    norm = acf[0] if acf[0] > 1e-12 else 1.0
    acf  = acf / norm

    # 搜索范围: 对应 [min_bpm, max_bpm] 的延迟
    lag_min = max(1, int(60.0 / max_bpm * fs))
    lag_max = min(n - 1, int(60.0 / min_bpm * fs))

    if lag_min >= lag_max:
        return 0.0

    search = acf[lag_min:lag_max]
    if len(search) < 3:
        return 0.0

    # 在搜索区间找第一个局部极大值(峰值显著性要求>0.1)
    peaks, props = find_peaks(search, prominence=0.08)
    if len(peaks) == 0:
        # 退化: 直接取最大值位置
        peak_lag = lag_min + int(np.argmax(search))
    else:
        # 取最显著的峰
        best = peaks[np.argmax(props['prominences'])]
        peak_lag = lag_min + int(best)

    if peak_lag <= 0:
        return 0.0

    period_s = peak_lag / fs
    bpm      = 60.0 / period_s
    if bpm < min_bpm or bpm > max_bpm:
        return 0.0
    return float(bpm)


# ════════════════════════════════════════════════════════════════
# 改进2: 多ROI ICA/PCA 信号融合
# ════════════════════════════════════════════════════════════════
def fuse_roi_signals_ica(roi_signals: np.ndarray, fs: float) -> np.ndarray:
    """
    多路ROI信号 (M, N) 经FastICA分离，选取在呼吸频段(0.1-0.5Hz)
    SNR最高的独立分量作为融合输出。
    roi_signals: (M, N) — M个ROI, 每ROI长度N帧
    返回: (N,) 最优独立分量（方向已校正为与均值信号正相关）
    """
    from sklearn.decomposition import FastICA
    from scipy.fft import fft, fftfreq

    M, N = roi_signals.shape
    if M == 1:
        return roi_signals[0]

    X = roi_signals.T                        # (N, M)
    n_comps = min(M, 5)

    try:
        ica = FastICA(n_components=n_comps, random_state=42,
                      max_iter=2000, tol=1e-3)
        sources = ica.fit_transform(X)       # (N, n_comps)
    except Exception:
        # ICA失败 → PCA回退
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1, random_state=42)
        src = pca.fit_transform(X).flatten()
        mean_sig = roi_signals.mean(axis=0)
        return src if np.dot(src, mean_sig) >= 0 else -src

    # 选呼吸频段SNR最高的分量
    freqs = fftfreq(N, 1.0 / fs)
    best_comp, best_snr = None, -999.0

    for k in range(n_comps):
        comp = sources[:, k]
        psd  = np.abs(fft(comp))[:N // 2]
        f_pos = freqs[:N // 2]
        in_band  = (f_pos >= 0.1) & (f_pos <= 0.5)
        out_band = ~in_band & (f_pos > 0)
        sig_pwr  = np.sum(psd[in_band]**2)
        noi_pwr  = np.sum(psd[out_band]**2)
        snr = 10 * np.log10(sig_pwr / noi_pwr) if noi_pwr > 0 else 20.0
        if snr > best_snr:
            best_snr  = snr
            best_comp = comp

    if best_comp is None:
        best_comp = sources[:, 0]

    # 方向对齐: 与ROI均值信号正相关
    mean_sig = roi_signals.mean(axis=0)
    if np.dot(best_comp, mean_sig) < 0:
        best_comp = -best_comp

    return best_comp


# ════════════════════════════════════════════════════════════════
# 数据加载（复用v2逻辑，独立实现不依赖v2文件）
# ════════════════════════════════════════════════════════════════
def _preprocess_frame(frame):
    f = frame.astype(np.float32)
    f = np.clip(f, 0, CLIP_MAX)
    f[f < DEADZONE] = 0.0
    f = median_filter(f, size=3)
    f = gaussian_filter(f, sigma=0.5)
    return f


def load_cushion_data(filepath):
    print(f"[DATA] {os.path.basename(filepath)}")
    frames, timestamps = [], []
    with open(filepath, 'r', encoding='utf-8') as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 1601:
                continue
            try:
                t  = datetime.strptime(parts[0], '%H:%M:%S.%f')
                ts = t.hour*3600 + t.minute*60 + t.second + t.microsecond/1e6
            except ValueError:
                continue
            raw = np.array(parts[1:1601], dtype=np.float32).reshape(40, 40)
            frames.append(_preprocess_frame(raw))
            timestamps.append(ts)

    frames = np.array(frames, dtype=np.float32)
    trim   = int(TRIM_SEC * FS)
    if len(frames) > 2 * trim:
        frames = frames[trim:-trim]
    print(f"       → {len(frames)} 帧 ({len(frames)/FS:.1f}s)")
    return frames


def load_reference_rsp(filepath):
    print(f"[REF]  {os.path.basename(filepath)}")
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

    trim = int(TRIM_SEC * fs_ref)
    if len(rsp_raw) > 2 * trim:
        rsp_raw = rsp_raw[trim:-trim]
    rsp_raw -= rsp_raw.mean()

    nyq = 0.5 * fs_ref
    b, a = butter(4, 1.0 / nyq, btype='low')
    rsp_lp = filtfilt(b, a, rsp_raw)
    ds = max(1, int(fs_ref / 10.0))
    rsp_ds = rsp_lp[::ds]
    fs_ds  = fs_ref / ds
    rsp_bp = butter_bandpass_filter(rsp_ds, 0.1, 0.5, fs=fs_ds, order=4)

    bpm_fpr  = calculate_bpm_fpr(rsp_bp, fs=fs_ds, min_dist_s=1.5)
    bpm_acr  = calculate_bpm_autocorr(rsp_bp, fs=fs_ds)
    print(f"       → 参考BPM  FPR={bpm_fpr:.2f}  AutoCorr={bpm_acr:.2f}")
    return rsp_bp, float(fs_ds), float(bpm_fpr), float(bpm_acr)


# ════════════════════════════════════════════════════════════════
# 改进3: 自适应左右分区（按压力重心，而非固定col=20）
# ════════════════════════════════════════════════════════════════
def find_lr_split(mean_frame: np.ndarray) -> int:
    """
    按列方向压力总和找压力最小的列作为左右分割线，
    比固定 col=20 更贴合实际坐姿偏移。
    约束在 [12, 28] 之间，防止极端值。
    """
    col_sum = mean_frame.sum(axis=0)          # (40,)
    # 在 [12, 28] 内找最小值列（分割线）
    search  = col_sum[12:28]
    split   = 12 + int(np.argmin(search))
    return split


def pick_roi_centers(mean_frame, k, min_dist, col_start, col_end):
    zone  = mean_frame[:, col_start:col_end]
    order = np.argsort(zone.ravel())[::-1]
    centers = []
    for idx in order:
        r, c_loc = np.unravel_index(idx, zone.shape)
        c = c_loc + col_start
        if not any(max(abs(r-cr), abs(c-cc)) < min_dist for cr, cc in centers):
            centers.append((r, c))
        if len(centers) == k:
            break
    while len(centers) < k:
        centers.append((zone.shape[0]//2, col_start + zone.shape[1]//2))
    return centers


def select_all_rois(frames: np.ndarray):
    mean_frame = frames.mean(axis=0)
    split = find_lr_split(mean_frame)
    print(f"[ROI]  自适应左右分割列 = {split}")

    left_centers  = pick_roi_centers(mean_frame, K_ROIS, MIN_DIST, 0,     split)
    right_centers = pick_roi_centers(mean_frame, K_ROIS, MIN_DIST, split, 40)

    rois = []
    for i, (r, c) in enumerate(left_centers):
        rois.append({'label': f'L{i+1}', 'center': (r, c), 'side': 'left'})
    for i, (r, c) in enumerate(right_centers):
        rois.append({'label': f'R{i+1}', 'center': (r, c), 'side': 'right'})

    for roi in rois:
        r, c = roi['center']
        print(f"       {roi['label']}: 行{r:2d} 列{c:2d}  "
              f"均值压力={mean_frame[r,c]:.1f}")
    return rois, mean_frame, split


def extract_roi_signal(frames, center) -> np.ndarray:
    """3×3 ROI均值时序 → 去均值 → 小波去噪 → 带通"""
    half = ROI_SIZE // 2
    r, c = center
    H, W = frames.shape[1], frames.shape[2]
    r_s, r_e = max(0, r-half), min(H, r+half+1)
    c_s, c_e = max(0, c-half), min(W, c+half+1)

    ts = frames[:, r_s:r_e, c_s:c_e].mean(axis=(1, 2))
    ts -= ts.mean()
    ts  = wavelet_denoise(ts, alpha=0.5)
    ts  = butter_bandpass_filter(ts, 0.1, 0.5, fs=FS, order=3)
    return ts.astype(np.float64)


def build_roi_signal_matrix(rois, frames) -> np.ndarray:
    """返回 (M, N) 矩阵：M=ROI数量, N=帧数"""
    sigs = [extract_roi_signal(frames, roi['center']) for roi in rois]
    return np.array(sigs, dtype=np.float64)


# ════════════════════════════════════════════════════════════════
# 改进4: 三种改进算法（双BPM估计方法对比）
# ════════════════════════════════════════════════════════════════
def _both_bpm(sig: np.ndarray) -> tuple:
    """同时用FPR和AutoCorr计算BPM，方便对比。"""
    bpm_fpr = calculate_bpm_fpr(sig, fs=FS, min_dist_s=1.5)
    bpm_acr = calculate_bpm_autocorr(sig, fs=FS)
    return float(bpm_fpr), float(bpm_acr)


def run_improved_acmd(roi_signals: np.ndarray, frames: np.ndarray) -> dict:
    """
    ACMD 改进:
      方案A: 单最优ROI (压力最大的那个)
      方案B: ICA融合全部ROI信号
    """
    # 方案A: 压力最大ROI
    energies = [np.std(s) for s in roi_signals]
    best_idx = int(np.argmax(energies))
    sig_a    = roi_signals[best_idx]
    out_a    = extract_breath_acmd(sig_a, fs=FS)
    bpm_a_fpr, bpm_a_acr = _both_bpm(out_a)

    # 方案B: ICA融合
    sig_b = fuse_roi_signals_ica(roi_signals, FS)
    out_b = extract_breath_acmd(sig_b, fs=FS)
    bpm_b_fpr, bpm_b_acr = _both_bpm(out_b)

    return {
        'A_单ROI_FPR':  bpm_a_fpr,
        'A_单ROI_ACR':  bpm_a_acr,
        'B_ICA融合_FPR': bpm_b_fpr,
        'B_ICA融合_ACR': bpm_b_acr,
        'sig_a': out_a,
        'sig_b': out_b,
    }


def run_improved_vmd(roi_signals: np.ndarray, frames: np.ndarray) -> dict:
    """
    VMD 改进:
      方案A: 单最优ROI
      方案B: ICA融合全部ROI信号
    """
    from algorithms.breath_extract import extract_breath_vmd

    energies = [np.std(s) for s in roi_signals]
    best_idx = int(np.argmax(energies))
    sig_a    = roi_signals[best_idx]
    out_a    = extract_breath_vmd(sig_a, fs=FS)
    bpm_a_fpr, bpm_a_acr = _both_bpm(out_a)

    sig_b = fuse_roi_signals_ica(roi_signals, FS)
    out_b = extract_breath_vmd(sig_b, fs=FS)
    bpm_b_fpr, bpm_b_acr = _both_bpm(out_b)

    return {
        'A_单ROI_FPR':  bpm_a_fpr,
        'A_单ROI_ACR':  bpm_a_acr,
        'B_ICA融合_FPR': bpm_b_fpr,
        'B_ICA融合_ACR': bpm_b_acr,
        'sig_a': out_a,
        'sig_b': out_b,
    }


def run_improved_goa_vmd(roi_signals: np.ndarray, frames: np.ndarray,
                          rois: list) -> dict:
    """
    GOA-VMD 改进:
      方案A: 原始全帧输入 (v2做法)
      方案B: 裁剪到左右最优ROI邻域子帧 (±ROI_CROP 像素)
      方案C: 直接用ICA融合信号替代帧级输入（降维后仍用GOA-VMD 1D路径）
    """
    ROI_CROP = 8    # 子帧邻域半径 (像素)
    H, W     = frames.shape[1], frames.shape[2]

    # 方案A: 全帧
    out_a = extract_breath_goa_vmd(frames, fs=FS)
    bpm_a_fpr, bpm_a_acr = _both_bpm(out_a)

    # 方案B: 子帧（取各ROI中心区域的并集bounding box）
    all_centers = [roi['center'] for roi in rois]
    r_min = max(0, min(rc[0] for rc in all_centers) - ROI_CROP)
    r_max = min(H, max(rc[0] for rc in all_centers) + ROI_CROP + 1)
    c_min = max(0, min(rc[1] for rc in all_centers) - ROI_CROP)
    c_max = min(W, max(rc[1] for rc in all_centers) + ROI_CROP + 1)
    sub_frames = frames[:, r_min:r_max, c_min:c_max]
    # 将子帧 pad 回 40×40 以满足算法内部假设
    padded = np.zeros_like(frames)
    padded[:, r_min:r_max, c_min:c_max] = sub_frames
    out_b = extract_breath_goa_vmd(padded, fs=FS)
    bpm_b_fpr, bpm_b_acr = _both_bpm(out_b)

    # 方案C: ICA融合信号走1D路径
    sig_c = fuse_roi_signals_ica(roi_signals, FS)
    out_c = extract_breath_goa_vmd(sig_c, fs=FS)   # 1D输入路径
    bpm_c_fpr, bpm_c_acr = _both_bpm(out_c)

    return {
        'A_全帧_FPR':   bpm_a_fpr,
        'A_全帧_ACR':   bpm_a_acr,
        'B_子帧_FPR':   bpm_b_fpr,
        'B_子帧_ACR':   bpm_b_acr,
        'C_ICA1D_FPR':  bpm_c_fpr,
        'C_ICA1D_ACR':  bpm_c_acr,
        'sig_a': out_a,
        'sig_b': out_b,
        'sig_c': out_c,
    }


# ════════════════════════════════════════════════════════════════
# 滑动窗口BPM（同时输出FPR和AutoCorr两条曲线）
# ════════════════════════════════════════════════════════════════
def sliding_bpm_dual(signal: np.ndarray, fs: float) -> dict:
    """返回 {'times', 'fpr', 'acr'} 三个数组。"""
    win  = int(WIN_SEC  * fs)
    step = int(STEP_SEC * fs)
    n    = len(signal)
    times, bpms_fpr, bpms_acr = [], [], []
    i = 0
    while i + win <= n:
        seg = signal[i:i+win]
        times.append((i + win/2) / fs)
        bpms_fpr.append(calculate_bpm_fpr(seg, fs=fs, min_dist_s=1.5))
        bpms_acr.append(calculate_bpm_autocorr(seg, fs=fs))
        i += step
    return {
        'times': np.array(times),
        'fpr':   np.array(bpms_fpr),
        'acr':   np.array(bpms_acr),
    }


# ════════════════════════════════════════════════════════════════
# 绘图：各算法改进方案的滑动窗口BPM曲线
# ════════════════════════════════════════════════════════════════
def plot_algo_comparison(algo_name: str,
                          variants: dict,
                          ref_sig: np.ndarray, ref_fs: float,
                          ref_bpm_fpr: float, ref_bpm_acr: float,
                          out_dir: str):
    """
    variants: {方案名: signal_array, ...}
    为每种算法画一张对比图，包含:
      - 参考RSP滑窗BPM (红实线FPR / 红虚线AutoCorr)
      - 每个方案的滑窗BPM (FPR实线 / AutoCorr虚线)
    """
    ref_sw = sliding_bpm_dual(ref_sig, ref_fs)

    fig, (ax_bpm, ax_sig) = plt.subplots(2, 1, figsize=(14, 9),
                                           constrained_layout=True)
    fig.suptitle(f'[{algo_name}] 改进方案对比  '
                 f'参考 FPR={ref_bpm_fpr:.2f} / ACR={ref_bpm_acr:.2f} BPM',
                 fontsize=13)

    # ── 滑窗BPM ──
    ax_bpm.plot(ref_sw['times'], ref_sw['fpr'], 'r-',  lw=2.2, label='参考-FPR',  zorder=10)
    ax_bpm.plot(ref_sw['times'], ref_sw['acr'], 'r--', lw=2.2, label='参考-AutoCorr', zorder=10)

    palette = plt.cm.tab10(np.linspace(0, 0.7, len(variants)))
    t_sig   = np.arange(next(iter(variants.values())).shape[0]) / FS

    for idx, (var_name, sig) in enumerate(variants.items()):
        sw = sliding_bpm_dual(sig, FS)
        c  = palette[idx]
        ax_bpm.plot(sw['times'], sw['fpr'], color=c, lw=1.6,
                    label=f'{var_name}-FPR')
        ax_bpm.plot(sw['times'], sw['acr'], color=c, lw=1.6, ls='--',
                    label=f'{var_name}-ACR')
        # 信号波形
        ax_sig.plot(t_sig, sig / (np.std(sig)+1e-9), color=c,
                    lw=1.1, alpha=0.8, label=var_name)

    ax_bpm.set_ylabel('BPM'); ax_bpm.set_xlabel('时间 (s)')
    ax_bpm.set_title('滑动窗口BPM曲线 (实=FPR, 虚=AutoCorr)')
    ax_bpm.legend(fontsize=7, ncol=3, loc='upper right')
    ax_bpm.grid(alpha=0.3)
    ax_bpm.set_ylim(0, 45)

    # 参考信号叠加
    t_ref = np.arange(len(ref_sig)) / ref_fs
    ax_sig.plot(t_ref, ref_sig / (np.std(ref_sig)+1e-9),
                'r-', lw=1.6, alpha=0.7, label='参考RSP(归一)')
    ax_sig.set_ylabel('归一化幅值'); ax_sig.set_xlabel('时间 (s)')
    ax_sig.set_title('提取信号波形（标准差归一化）')
    ax_sig.legend(fontsize=7, ncol=3, loc='upper right')
    ax_sig.grid(alpha=0.3)

    safe = algo_name.replace(' ', '_')
    out_path = os.path.join(out_dir, f'改进_{safe}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {algo_name} → {out_path}")


# ════════════════════════════════════════════════════════════════
# CSV结果汇总
# ════════════════════════════════════════════════════════════════
def save_summary_csv(results: dict, ref_bpm_fpr: float,
                     ref_bpm_acr: float, out_dir: str):
    """
    results 结构:
      {'ACMD': {...方案键: bpm值...}, 'VMD': {...}, 'GOA-VMD': {...}}
    """
    path = os.path.join(out_dir, '改进结果汇总.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['算法', '方案', 'BPM', '参考BPM(FPR)',
                    '误差(FPR)', '参考BPM(ACR)', '误差(ACR)'])

        for algo_name, variants in results.items():
            for var_name, bpm in variants.items():
                if not isinstance(bpm, float):
                    continue           # 跳过signal数组字段
                is_acr = 'ACR' in var_name
                ref    = ref_bpm_acr if is_acr else ref_bpm_fpr
                err    = abs(bpm - ref)
                if is_acr:
                    w.writerow([algo_name, var_name, f'{bpm:.3f}',
                                 '', '', f'{ref:.3f}', f'{err:.3f}'])
                else:
                    w.writerow([algo_name, var_name, f'{bpm:.3f}',
                                 f'{ref:.3f}', f'{err:.3f}', '', ''])

    print(f"  CSV汇总 → {path}")


# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print('\n' + '='*60)
    print('  呼吸算法改进版 v3.0')
    print(f'  输出目录: {OUT_DIR}')
    print('='*60)

    # 1. 数据加载
    frames = load_cushion_data(DATA_FILE)
    ref_sig, ref_fs, ref_bpm_fpr, ref_bpm_acr = load_reference_rsp(REF_FILE)

    # 2. ROI选取 & 信号矩阵
    rois, mean_frame, split = select_all_rois(frames)
    roi_mat = build_roi_signal_matrix(rois, frames)   # (8, N)
    print(f"       ROI信号矩阵: {roi_mat.shape}")

    # 3. 运行三种改进算法
    print('\n[ALGO] 运行改进算法...')
    print('  ACMD...')
    acmd_res = run_improved_acmd(roi_mat, frames)
    print('  VMD...')
    vmd_res  = run_improved_vmd(roi_mat, frames)
    print('  GOA-VMD...')
    goa_res  = run_improved_goa_vmd(roi_mat, frames, rois)

    # 4. 控制台汇总
    print('\n' + '='*60)
    print(f'  参考: FPR={ref_bpm_fpr:.2f}  AutoCorr={ref_bpm_acr:.2f} BPM')
    print(f'  {"方案":<22}  {"FPR-BPM":>9}  {"FPR-err":>8}  '
          f'{"ACR-BPM":>9}  {"ACR-err":>8}')
    print(f'  {"-"*62}')

    def _row(label, fpr_key, acr_key, d):
        b_f = d[fpr_key]; b_a = d[acr_key]
        e_f = abs(b_f - ref_bpm_fpr)
        e_a = abs(b_a - ref_bpm_acr)
        flag_f = 'OK' if e_f<=1.5 else ('~' if e_f<=3 else 'X')
        flag_a = 'OK' if e_a<=1.5 else ('~' if e_a<=3 else 'X')
        print(f'  {label:<22}  {b_f:>9.2f}  {e_f:>7.2f}{flag_f}  '
              f'{b_a:>9.2f}  {e_a:>7.2f}{flag_a}')

    print('  [ACMD]')
    _row('A-单ROI',    'A_单ROI_FPR',   'A_单ROI_ACR',   acmd_res)
    _row('B-ICA融合',  'B_ICA融合_FPR', 'B_ICA融合_ACR', acmd_res)
    print('  [VMD]')
    _row('A-单ROI',    'A_单ROI_FPR',   'A_单ROI_ACR',   vmd_res)
    _row('B-ICA融合',  'B_ICA融合_FPR', 'B_ICA融合_ACR', vmd_res)
    print('  [GOA-VMD]')
    _row('A-全帧',    'A_全帧_FPR',  'A_全帧_ACR',  goa_res)
    _row('B-子帧',    'B_子帧_FPR',  'B_子帧_ACR',  goa_res)
    _row('C-ICA_1D', 'C_ICA1D_FPR', 'C_ICA1D_ACR', goa_res)
    print('='*60)

    # 5. 绘图
    print('\n[PLOT] 生成图表...')
    plot_algo_comparison(
        'ACMD',
        {'A单ROI': acmd_res['sig_a'], 'B-ICA融合': acmd_res['sig_b']},
        ref_sig, ref_fs, ref_bpm_fpr, ref_bpm_acr, OUT_DIR
    )
    plot_algo_comparison(
        'VMD',
        {'A-单ROI': vmd_res['sig_a'], 'B-ICA融合': vmd_res['sig_b']},
        ref_sig, ref_fs, ref_bpm_fpr, ref_bpm_acr, OUT_DIR
    )
    plot_algo_comparison(
        'GOA-VMD',
        {'A-全帧': goa_res['sig_a'], 'B-子帧': goa_res['sig_b'],
         'C-ICA_1D': goa_res['sig_c']},
        ref_sig, ref_fs, ref_bpm_fpr, ref_bpm_acr, OUT_DIR
    )

    # 6. CSV
    print('\n[CSV]  保存结果...')
    results = {'ACMD': acmd_res, 'VMD': vmd_res, 'GOA-VMD': goa_res}
    save_summary_csv(results, ref_bpm_fpr, ref_bpm_acr, OUT_DIR)

    print(f'\n完成！→ {OUT_DIR}\n')


if __name__ == '__main__':
    main()

