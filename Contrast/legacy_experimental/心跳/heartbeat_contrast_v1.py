# -*- coding: utf-8 -*-
"""
心跳提取算法对比脚本 heartbeat_contrast_v1.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
流程:
  1. 加载40×40座垫数据  → 预处理 → ROI → ICA融合
  2. VME基线去除(呼吸漂移) → 带通0.8–2.2Hz
  3. 运行全部心跳提取算法
  4. BPM估计(FFT主频法 + FPR峰值法 → 取最优)
  5. 对比参考CH2(PPG, 2000Hz) → 误差报告

最优策略(从呼吸分析迭代结论迁移):
  - 多ROI ICA融合信号作为算法输入
  - VME剥离呼吸基线
  - FFT-BPM 作为主要估计器(在低采样率11.2Hz下最稳定)
  - FPR作为辅助验证
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys, os, csv, time, warnings
warnings.filterwarnings('ignore')

_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_DIR))
sys.path.insert(0, os.path.join(ROOT, '40_40_Extraction_1'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime
from scipy.ndimage import median_filter, gaussian_filter
from scipy.signal import butter, filtfilt, find_peaks, correlate

from algorithms.base import (
    butter_bandpass_filter, wavelet_denoise,
)
from algorithms.heartbeat_extract import (
    extract_heartbeat_mean,
    extract_heartbeat_acmd,
    extract_heartbeat_vmd,
    extract_heartbeat_emd,
    extract_heartbeat_vme,
)

# ── 配置 ─────────────────────────────────────────────────────────
FS       = 11.2       # 座垫采样率
TRIM_SEC = 20.0
ROI_SIZE = 3
K_ROIS   = 4
MIN_DIST = 5
WIN_SEC  = 30.0
STEP_SEC = 5.0
DEADZONE = 30
CLIP_MAX = 2000

# 心跳频段
HB_LOW  = 0.8    # Hz
HB_HIGH = 2.2    # Hz (对应最高约132 BPM)
BPM_MIN = 40.0
BPM_MAX = 150.0

DATA_FILE = os.path.join(ROOT, 'data',         '20260702_160410_40x40.txt')
REF_FILE  = os.path.join(ROOT, 'Precise_Data', '刘若红0702.txt')
OUT_DIR   = os.path.join(ROOT, 'Contrast', '心跳', '刘若红_0702_160410_心跳对比_v1')

ALGO_NAMES = ['均值法', 'ACMD', 'VMD', 'EMD', 'VME']


def _font():
    for n in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']:
        try:
            fm.findfont(fm.FontProperties(family=n), fallback_to_default=False)
            plt.rcParams.update({'font.family': n, 'axes.unicode_minus': False})
            return
        except Exception:
            pass
    plt.rcParams['axes.unicode_minus'] = False

_font()


# ════════════════════════════════════════════════════════════════
# BPM 估计器 (心跳版)
# ════════════════════════════════════════════════════════════════
def bpm_fft(sig, fs, min_bpm=BPM_MIN, max_bpm=BPM_MAX):
    """FFT主频法: 在0.8–2.2Hz找PSD最大峰"""
    n     = len(sig)
    freqs = np.fft.rfftfreq(n, 1.0/fs)
    psd   = np.abs(np.fft.rfft(sig - sig.mean()))**2
    mask  = (freqs >= min_bpm/60) & (freqs <= max_bpm/60)
    if not mask.any():
        return 0.0
    return float(freqs[mask][np.argmax(psd[mask])] * 60)


def bpm_fpr(sig, fs, min_bpm=BPM_MIN, max_bpm=BPM_MAX):
    """FPR峰值计数法: min_dist=0.4s (最快心跳约150BPM→0.4s间隔)"""
    min_dist = max(1, int(fs * 60 / max_bpm))
    peaks, _ = find_peaks(sig, distance=min_dist,
                          prominence=(np.ptp(sig) * 0.1) if np.ptp(sig) > 0 else 0.01)
    if len(peaks) < 2:
        return 0.0
    bpm = 60 * fs / np.mean(np.diff(peaks))
    return float(bpm) if min_bpm <= bpm <= max_bpm else 0.0


def best_bpm(sig, fs):
    """
    取 FFT 和 FPR 两种估计的最优值:
    优先选与生理均值(~75BPM)更接近的那个,
    若两者均有效则取绝对值差最小的。
    """
    f = bpm_fft(sig, fs)
    p = bpm_fpr(sig, fs)
    ref = 75.0   # 正常人静息心率参考
    valid = [v for v in [f, p] if BPM_MIN <= v <= BPM_MAX]
    if not valid:
        return 0.0, f, p
    best = min(valid, key=lambda v: abs(v - ref))
    return float(best), float(f), float(p)


# ════════════════════════════════════════════════════════════════
# 数据加载
# ════════════════════════════════════════════════════════════════
def load_cushion(fp):
    print(f"[DATA] {os.path.basename(fp)}")
    frames = []
    with open(fp, 'r', encoding='utf-8') as fh:
        for line in fh:
            p = line.split()
            if len(p) < 1601: continue
            try:
                datetime.strptime(p[0], '%H:%M:%S.%f')
            except ValueError:
                continue
            raw = np.array(p[1:1601], dtype=np.float32).reshape(40, 40)
            f   = np.clip(raw, 0, CLIP_MAX).astype(np.float32)
            f[f < DEADZONE] = 0
            f = median_filter(f, size=3)
            f = gaussian_filter(f, sigma=0.5)
            frames.append(f)
    frames = np.array(frames, dtype=np.float32)
    trim   = int(TRIM_SEC * FS)
    if len(frames) > 2*trim:
        frames = frames[trim:-trim]
    print(f"       {len(frames)} 帧 ({len(frames)/FS:.1f}s)")
    return frames


def load_ref_ch2(fp):
    """
    解析 ACQ txt，提取 CH2 (PPG) 信号并计算参考心率。
    CH2 = PPG (光电容积脉搏波)，2000Hz，峰值直接对应心跳周期。
    处理流程: 去头尾20s → 低通5Hz → 下采样50Hz → 带通0.8-2.2Hz → FPR/FFT
    """
    print(f"[REF]  {os.path.basename(fp)}")
    fs_r = 2000.0
    with open(fp, 'r', encoding='utf-8', errors='replace') as fh:
        lines = fh.readlines()
    for ln in lines[:4]:
        if 'msec/sample' in ln:
            try: fs_r = 1000.0 / float(ln.strip().split()[0])
            except: pass
            break
    # 定位数据起始行
    di = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith('CH1'): di = i+2; break
    raw = []
    for ln in lines[di:]:
        cols = ln.strip().split('\t')
        try:
            if len(cols) >= 2:
                raw.append(float(cols[1]))   # CH2 是第2列
        except (ValueError, IndexError):
            continue
    raw = np.array(raw, dtype=np.float64)
    print(f"       CH2(PPG): {len(raw)} 点 @ {fs_r:.0f}Hz ({len(raw)/fs_r:.1f}s)")

    trim = int(TRIM_SEC * fs_r)
    if len(raw) > 2*trim: raw = raw[trim:-trim]
    raw -= raw.mean()

    # 低通抗混叠 5Hz → 下采样至50Hz
    b, a  = butter(4, 5.0/(0.5*fs_r), btype='low')
    lp    = filtfilt(b, a, raw)
    ds    = max(1, int(fs_r / 50.0))
    sig   = lp[::ds];  fs_ds = fs_r / ds
    sig   = butter_bandpass_filter(sig, HB_LOW, HB_HIGH, fs=fs_ds, order=4)

    ref, f, p = best_bpm(sig, fs_ds)
    print(f"       参考心率: FFT={f:.1f}  FPR={p:.1f}  最优={ref:.1f} BPM")
    return sig, float(fs_ds), float(ref), float(f), float(p)


# ════════════════════════════════════════════════════════════════
# ROI 选取 + ICA 融合 + VME 基线去除
# ════════════════════════════════════════════════════════════════
def poly_detrend(sig, order=3):
    t = np.arange(len(sig), dtype=np.float64)
    return sig - np.polyval(np.polyfit(t, sig, order), t)


def _split_col(mf):
    return 12 + int(np.argmin(mf.sum(axis=0)[12:28]))


def _pick_centers(zone, k, md, c_off):
    order = np.argsort(zone.ravel())[::-1]
    cens  = []
    for idx in order:
        r, cl = np.unravel_index(idx, zone.shape)
        c = cl + c_off
        if not any(max(abs(r-cr), abs(c-cc)) < md for cr,cc in cens):
            cens.append((r, c))
        if len(cens) == k: break
    while len(cens) < k:
        cens.append((zone.shape[0]//2, c_off + zone.shape[1]//2))
    return cens


def _ica_fuse(roi_mat, fs):
    from sklearn.decomposition import FastICA, PCA
    from scipy.fft import fft as _fft, fftfreq as _ff
    M, N = roi_mat.shape
    if M == 1: return roi_mat[0]
    X  = roi_mat.T
    nc = min(M, 5)
    try:
        src = FastICA(n_components=nc, random_state=42,
                      max_iter=2000, tol=1e-3).fit_transform(X)
    except Exception:
        src = PCA(n_components=nc, random_state=42).fit_transform(X)
    freqs = _ff(N, 1.0/fs)[:N//2]
    # 选心跳频段 SNR 最高的独立分量
    inb  = (freqs >= HB_LOW) & (freqs <= HB_HIGH)
    best, bsnr = None, -999.0
    for k in range(src.shape[1]):
        c   = src[:, k]
        psd = np.abs(_fft(c))[:N//2]**2
        snr = 10*np.log10(psd[inb].sum() /
                          (psd[~inb & (freqs>0)].sum() + 1e-9))
        if snr > bsnr: bsnr, best = snr, c
    ms = roi_mat.mean(axis=0)
    return best if np.dot(best, ms) >= 0 else -best


def _vme_remove_breath(sig, fs):
    """
    用 VME_Core 提取呼吸基线漂移分量（~0.25Hz）并剔除，
    参考 HeartbeatRate/algorithms/base.py 中的实现。
    """
    from algorithms.base import VME_Core
    n = len(sig)
    if n < 20:
        return sig
    # 估算当前呼吸主频（用于 VME 的 f_init）
    freqs = np.fft.rfftfreq(n, 1.0/fs)
    psd   = np.abs(np.fft.rfft(sig - sig.mean()))**2
    bd_mask = (freqs >= 0.1) & (freqs <= 0.5)
    f_breath = (freqs[bd_mask][np.argmax(psd[bd_mask])]
                if bd_mask.any() else 0.25)
    # alpha 自适应: 心跳频率偏低时加大 VME alpha 以更彻底去除
    f_heart_est = bpm_fft(sig, fs) / 60.0
    if 0 < f_heart_est <= 1.25:
        alpha_bd = 1000.0 * np.exp(1.09 * ((f_heart_est - 1.25) / -0.5)**2)
    else:
        alpha_bd = 1000.0
    try:
        u_bd = VME_Core(sig - sig.mean(), fs=fs,
                        f_init=f_breath, alpha=alpha_bd)
        return sig - u_bd
    except Exception:
        return sig


def build_fused_signal(frames):
    """
    ROI选取 → poly_detrend → ICA融合 → VME去呼吸基线 → 带通0.8-2.2Hz
    返回已准备好输入心跳算法的1D信号
    """
    mf   = frames.mean(axis=0)
    sp   = _split_col(mf)
    lc   = _pick_centers(mf[:, :sp],  K_ROIS, MIN_DIST, 0)
    rc   = _pick_centers(mf[:, sp:],  K_ROIS, MIN_DIST, sp)
    rois = ([{'label': f'L{i+1}', 'c': c} for i, c in enumerate(lc)] +
            [{'label': f'R{i+1}', 'c': c} for i, c in enumerate(rc)])

    half = ROI_SIZE // 2
    H, W = frames.shape[1], frames.shape[2]
    sigs = []
    for roi in rois:
        r, c = roi['c']
        rs, re = max(0, r-half), min(H, r+half+1)
        cs, ce = max(0, c-half), min(W, c+half+1)
        ts = frames[:, rs:re, cs:ce].mean(axis=(1, 2))
        ts = poly_detrend(ts, order=3)
        ts = wavelet_denoise(ts, alpha=0.3)    # 心跳用更小 alpha
        sigs.append(ts.astype(np.float64))
        print(f"       {roi['label']}: 行{r:2d}列{c:2d}  压力={mf[r,c]:.1f}")

    roi_mat = np.array(sigs)
    fused   = _ica_fuse(roi_mat, FS)

    # VME 去呼吸基线
    fused = _vme_remove_breath(fused, FS)
    # 带通
    fused = butter_bandpass_filter(fused, HB_LOW, HB_HIGH, fs=FS, order=4)
    print(f"       ICA+VME融合信号 shape={fused.shape}")
    return fused, rois, mf


# ════════════════════════════════════════════════════════════════
# 算法执行（ICA+VME融合信号输入 + FFT/FPR最优BPM + 计时）
# ════════════════════════════════════════════════════════════════
def run_all(fused: np.ndarray, frames: np.ndarray) -> dict:
    algo_map = {
        '均值法': lambda s, f: extract_heartbeat_mean(s, fs=FS),
        'ACMD':   lambda s, f: extract_heartbeat_acmd(s, fs=FS),
        'VMD':    lambda s, f: extract_heartbeat_vmd (s, fs=FS),
        'EMD':    lambda s, f: extract_heartbeat_emd (s, fs=FS),
        'VME':    lambda s, f: extract_heartbeat_vme (s, fs=FS),
    }
    print('\n[ALGO]  (输入=ICA+VME融合信号, BPM=FFT/FPR最优)')
    results = {}
    for name in ALGO_NAMES:
        print(f"    {name}...", end='', flush=True)
        t0 = time.perf_counter()
        try:
            sig = algo_map[name](fused, frames)
            # 算法输出可能长度不一致，裁剪/补齐
            sig = np.array(sig, dtype=np.float64).flatten()
            if len(sig) != len(fused):
                sig = sig[:len(fused)] if len(sig) > len(fused) \
                      else np.pad(sig, (0, len(fused)-len(sig)))
            bpm, f_val, p_val = best_bpm(sig, FS)
        except Exception as e:
            print(f" [err:{e}]")
            sig = np.zeros_like(fused); bpm = f_val = p_val = 0.0
        elapsed = (time.perf_counter() - t0) * 1000
        results[name] = {
            'sig': sig, 'bpm': float(bpm),
            'fft': float(f_val), 'fpr': float(p_val),
            'time_ms': float(elapsed),
        }
        print(f"  FFT={f_val:.1f} FPR={p_val:.1f} 最优={bpm:.1f} BPM  耗时={elapsed:.1f}ms")
    return results


# ════════════════════════════════════════════════════════════════
# 滑动窗口 BPM
# ════════════════════════════════════════════════════════════════
def sw_bpm(sig, fs):
    win, step = int(WIN_SEC*fs), int(STEP_SEC*fs)
    T, B = [], []
    i = 0
    while i + win <= len(sig):
        seg = sig[i:i+win]
        b, _, _ = best_bpm(seg, fs)
        T.append((i + win/2) / fs)
        B.append(b)
        i += step
    return np.array(T), np.array(B)


# ════════════════════════════════════════════════════════════════
# 单算法独立图
# ════════════════════════════════════════════════════════════════
def plot_one(name, res, ref_sig, ref_fs, ref_bpm, out_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7),
                                    constrained_layout=True)
    err = abs(res['bpm'] - ref_bpm)
    fig.suptitle(
        f'[{name}]   BPM={res["bpm"]:.1f}  |  参考={ref_bpm:.1f}  |  '
        f'误差={err:.1f} BPM  |  耗时={res["time_ms"]:.1f}ms',
        fontsize=11, fontweight='bold'
    )
    # 面板1: 滑窗BPM
    rT, rB = sw_bpm(ref_sig, ref_fs)
    aT, aB = sw_bpm(res['sig'], FS)
    ax1.plot(rT, rB, 'r-', lw=2.2, label=f'参考PPG ({ref_bpm:.1f} BPM)')
    ax1.plot(aT, aB, color='#27ae60', lw=1.8, label=f'{name} ({res["bpm"]:.1f} BPM)')
    ax1.axhline(ref_bpm, color='red', lw=0.9, ls='--', alpha=0.5)
    ax1.set_ylabel('BPM');  ax1.set_xlabel('时间 (s)')
    ax1.set_title(f'滑动窗口BPM  (窗口={WIN_SEC:.0f}s / 步长={STEP_SEC:.0f}s)')
    ax1.legend(fontsize=10);  ax1.grid(alpha=0.25)
    yc = ref_bpm if ref_bpm > 0 else 75
    ax1.set_ylim(max(0, yc - 30), yc + 30)

    # 面板2: 信号波形
    def _n(s): sd=np.std(s); return s/sd if sd>1e-9 else s
    t_r = np.arange(len(ref_sig)) / ref_fs
    t_a = np.arange(len(res['sig'])) / FS
    ax2.plot(t_r, _n(ref_sig),    'r-',            lw=1.5, alpha=0.8, label='参考PPG')
    ax2.plot(t_a, _n(res['sig']), color='#2980b9', lw=1.2, alpha=0.8, label=f'ICA+VME+{name}')
    ax2.set_ylabel('归一化幅值');  ax2.set_xlabel('时间 (s)')
    ax2.set_title('提取信号波形（标准差归一化）')
    ax2.legend(fontsize=9);  ax2.grid(alpha=0.25);  ax2.set_ylim(-5, 5)

    safe = name.replace(' ', '_').replace('-', '_')
    path = os.path.join(out_dir, f'{safe}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {name:<12} → {os.path.basename(path)}")


# ════════════════════════════════════════════════════════════════
# 汇总图：误差 + 处理时间双坐标
# ════════════════════════════════════════════════════════════════
def plot_summary(all_res, ref_bpm, out_dir):
    names  = list(all_res.keys())
    errors = [abs(all_res[n]['bpm'] - ref_bpm) for n in names]
    times  = [all_res[n]['time_ms']             for n in names]

    fig, ax1 = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax2 = ax1.twinx()
    x, w = np.arange(len(names)), 0.35

    c_err = ['#27ae60' if e<=5 else '#f39c12' if e<=10 else '#e74c3c' for e in errors]
    b1 = ax1.bar(x-w/2, errors, w, color=c_err,    alpha=0.88, label='BPM误差 (左轴)')
    b2 = ax2.bar(x+w/2, times,  w, color='#5b9bd5', alpha=0.72, label='耗时ms (右轴)')

    for bar, v in zip(b1, errors):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    for bar, v in zip(b2, times):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                 f'{v:.0f}', ha='center', va='bottom', fontsize=8, color='#2e6da4')

    ax1.axhline(5,  color='green',  lw=1.2, ls='--', alpha=0.7, label='±5 BPM')
    ax1.axhline(10, color='orange', lw=1.2, ls='--', alpha=0.7, label='±10 BPM')
    ax1.set_xticks(x);  ax1.set_xticklabels(names, fontsize=11)
    ax1.set_ylabel('|BPM误差|'); ax2.set_ylabel('耗时 (ms)', color='#2e6da4')
    ax1.set_title(f'心跳算法对比 (参考PPG={ref_bpm:.1f} BPM, ICA+VME+最优BPM)')
    lines1, lbs1 = ax1.get_legend_handles_labels()
    lines2, lbs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, lbs1+lbs2, fontsize=8, loc='upper left')
    ax1.grid(alpha=0.25, axis='y')

    path = os.path.join(out_dir, '汇总误差与耗时.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  汇总图 → {path}")


# ════════════════════════════════════════════════════════════════
# CSV
# ════════════════════════════════════════════════════════════════
def save_csv(all_res, ref_bpm, out_dir):
    path = os.path.join(out_dir, '心跳结果汇总.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['算法', '最优BPM', 'FFT-BPM', 'FPR-BPM',
                    '参考BPM', '绝对误差', '相对误差(%)', '耗时(ms)', '策略'])
        for nm, r in all_res.items():
            err = abs(r['bpm'] - ref_bpm)
            rel = err/ref_bpm*100 if ref_bpm > 0 else float('nan')
            w.writerow([nm,
                f"{r['bpm']:.2f}", f"{r['fft']:.2f}", f"{r['fpr']:.2f}",
                f"{ref_bpm:.2f}", f"{err:.2f}", f"{rel:.1f}",
                f"{r['time_ms']:.1f}", 'ICA+VME+最优BPM'])
    print(f"  CSV → {path}")


# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print('\n'+'='*62)
    print('  心跳提取算法对比 v1.0  (ICA+VME+最优BPM策略)')
    print(f'  输出: {OUT_DIR}')
    print('='*62)

    frames = load_cushion(DATA_FILE)
    ref_sig, ref_fs, ref_bpm, ref_f, ref_p = load_ref_ch2(REF_FILE)

    print('\n[ROI+ICA+VME]')
    t0 = time.perf_counter()
    fused, rois, mf = build_fused_signal(frames)
    pre_ms = (time.perf_counter()-t0)*1000
    print(f"       预处理耗时: {pre_ms:.0f}ms")

    all_res = run_all(fused, frames)

    print('\n[PLOT] 单算法图:')
    for name in ALGO_NAMES:
        plot_one(name, all_res[name], ref_sig, ref_fs, ref_bpm, OUT_DIR)

    print('\n[PLOT] 汇总图:')
    plot_summary(all_res, ref_bpm, OUT_DIR)

    print('\n[CSV]')
    save_csv(all_res, ref_bpm, OUT_DIR)

    print('\n'+'='*62)
    print(f'  参考心率 (CH2 PPG): FFT={ref_f:.1f}  FPR={ref_p:.1f}  最优={ref_bpm:.1f} BPM')
    print(f'  {"算法":<12}  {"最优BPM":>8}  {"误差":>7}  {"耗时ms":>9}')
    print(f'  {"-"*44}')
    for nm in ALGO_NAMES:
        r   = all_res[nm]
        err = abs(r['bpm'] - ref_bpm)
        flg = 'OK' if err<=5 else ('~' if err<=10 else 'X')
        print(f'  {nm:<12}  {r["bpm"]:>8.1f}  {err:>7.1f}  {r["time_ms"]:>9.1f}  {flg}')
    print('='*62+'\n')
    print(f'完成 → {OUT_DIR}\n')


if __name__ == '__main__':
    main()
