# -*- coding: utf-8 -*-
"""
心跳提取算法最终对比 heartbeat_contrast_final.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
基于 v1~v4 迭代实验结论，对每种算法独立选取最优策略:

┌──────────────┬────────────────────┬──────────┐
│ 算法         │ 最优输入           │ BPM估计  │
├──────────────┼────────────────────┼──────────┤
│ VMD          │ 单最优ROI          │ FPR      │
│ EMD          │ ICA融合            │ FFT      │
│ ACMD         │ ICA融合            │ FPR      │
│ 均值法       │ ICA融合            │ ACR      │
│ VME          │ ICA融合            │ FPR      │
└──────────────┴────────────────────┴──────────┘

同时输出:
  - 各版本最优结果对比汇总表
  - 每种算法独立波形图
  - 汇总误差+耗时图
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

from algorithms.base import butter_bandpass_filter, wavelet_denoise
from algorithms.heartbeat_extract import (
    extract_heartbeat_mean, extract_heartbeat_acmd,
    extract_heartbeat_vmd,  extract_heartbeat_emd,
    extract_heartbeat_vme,
)

FS       = 11.2
TRIM_SEC = 20.0
ROI_SIZE = 3
K_ROIS   = 4
MIN_DIST = 5
WIN_SEC  = 25.0
STEP_SEC = 5.0
DEADZONE = 30
CLIP_MAX = 2000
HB_LOW, HB_HIGH = 0.8, 2.2
BPM_MIN, BPM_MAX = 40.0, 150.0

DATA_FILE = os.path.join(ROOT, 'data',         '20260702_160410_40x40.txt')
REF_FILE  = os.path.join(ROOT, 'Precise_Data', '刘若红0702.txt')
OUT_DIR   = os.path.join(ROOT, 'Contrast', '心跳', '刘若红_0702_160410_心跳最终版')
ALGO_NAMES = ['均值法', 'ACMD', 'VMD', 'EMD', 'VME']


def _font():
    for n in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']:
        try:
            fm.findfont(fm.FontProperties(family=n), fallback_to_default=False)
            plt.rcParams.update({'font.family': n, 'axes.unicode_minus': False})
            return
        except: pass
    plt.rcParams['axes.unicode_minus'] = False
_font()


# ════════════════════════════════════════════════════════════════
# BPM 估计器（三种，供各算法按需选用）
# ════════════════════════════════════════════════════════════════
def bpm_fpr(sig, fs):
    md = max(1, int(fs * 60 / BPM_MAX))
    pks, _ = find_peaks(sig, distance=md,
                        prominence=np.ptp(sig)*0.1 if np.ptp(sig) > 0 else 0.01)
    if len(pks) < 2: return 0.0
    b = 60 * fs / np.mean(np.diff(pks))
    return float(b) if BPM_MIN <= b <= BPM_MAX else 0.0


def bpm_fft(sig, fs):
    n = len(sig)
    freqs = np.fft.rfftfreq(n, 1.0/fs)
    psd   = np.abs(np.fft.rfft(sig - sig.mean()))**2
    mask  = (freqs >= BPM_MIN/60) & (freqs <= BPM_MAX/60)
    if not mask.any(): return 0.0
    return float(freqs[mask][np.argmax(psd[mask])] * 60)


def bpm_acr(sig, fs):
    n = len(sig)
    if n < 30: return 0.0
    s   = sig - sig.mean()
    acf = correlate(s, s, mode='full')[n-1:]
    acf = acf / (acf[0] + 1e-12)
    lmin = max(1, int(60.0/BPM_MAX * fs))
    lmax = min(n-1, int(60.0/BPM_MIN * fs))
    if lmin >= lmax: return 0.0
    seg = acf[lmin:lmax]
    pks, pr = find_peaks(seg, prominence=0.05)
    pk = (lmin + int(pks[np.argmax(pr['prominences'])])
          if len(pks) else lmin + int(np.argmax(seg)))
    b = 60.0 / (pk / fs)
    return float(b) if BPM_MIN <= b <= BPM_MAX else 0.0


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
            try: datetime.strptime(p[0], '%H:%M:%S.%f')
            except ValueError: continue
            raw = np.array(p[1:1601], dtype=np.float32).reshape(40, 40)
            f   = np.clip(raw, 0, CLIP_MAX).astype(np.float32)
            f[f < DEADZONE] = 0
            f = median_filter(f, size=3)
            f = gaussian_filter(f, sigma=0.5)
            frames.append(f)
    frames = np.array(frames, dtype=np.float32)
    trim   = int(TRIM_SEC * FS)
    if len(frames) > 2*trim: frames = frames[trim:-trim]
    print(f"       {len(frames)} 帧 ({len(frames)/FS:.1f}s)")
    return frames


def load_ref_ch2(fp):
    print(f"[REF]  {os.path.basename(fp)}")
    fs_r = 2000.0
    with open(fp, 'r', encoding='utf-8', errors='replace') as fh:
        lines = fh.readlines()
    for ln in lines[:4]:
        if 'msec/sample' in ln:
            try: fs_r = 1000.0/float(ln.strip().split()[0])
            except: pass
            break
    di = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith('CH1'): di = i+2; break
    ch1r, ch2r = [], []
    for ln in lines[di:]:
        cols = ln.strip().split('\t')
        try: ch1r.append(float(cols[0])); ch2r.append(float(cols[1]))
        except: continue

    # 呼吸基频（CH1）
    ch1 = np.array(ch1r, dtype=np.float64)
    trim = int(TRIM_SEC * fs_r)
    if len(ch1) > 2*trim: ch1 = ch1[trim:-trim]
    ch1 -= ch1.mean()
    b, a = butter(4, 1.0/(0.5*fs_r), btype='low')
    ch1_ds = filtfilt(b, a, ch1)[::max(1, int(fs_r/10))]
    fs_rsp = fs_r / max(1, int(fs_r/10))
    ch1_bp = butter_bandpass_filter(ch1_ds, 0.1, 0.5, fs=fs_rsp, order=4)
    fr = np.fft.rfftfreq(len(ch1_bp), 1.0/fs_rsp)
    ps = np.abs(np.fft.rfft(ch1_bp))**2
    rm = (fr >= 0.1) & (fr <= 0.5)
    breath_freq = float(fr[rm][np.argmax(ps[rm])]) if rm.any() else 0.238
    print(f"       呼吸={breath_freq*60:.1f}BPM  6次谐波={breath_freq*6*60:.1f}BPM")

    # 参考心率（CH2 PPG）
    ch2 = np.array(ch2r, dtype=np.float64)
    if len(ch2) > 2*trim: ch2 = ch2[trim:-trim]
    ch2 -= ch2.mean()
    b2, a2 = butter(4, 5.0/(0.5*fs_r), btype='low')
    ds = max(1, int(fs_r/50))
    ch2_ds = filtfilt(b2, a2, ch2)[::ds]
    fs_ppg = fs_r / ds
    ch2_bp = butter_bandpass_filter(ch2_ds, HB_LOW, HB_HIGH, fs=fs_ppg, order=4)
    f_ = bpm_fft(ch2_bp, fs_ppg)
    a_ = bpm_acr(ch2_bp, fs_ppg)
    p_ = bpm_fpr(ch2_bp, fs_ppg)
    ref = float(np.median([v for v in [f_, a_, p_] if BPM_MIN <= v <= BPM_MAX]))
    print(f"       参考心率: FFT={f_:.1f} ACR={a_:.1f} FPR={p_:.1f} → 中位={ref:.1f}")
    return ch2_bp, float(fs_ppg), float(ref), float(breath_freq)


# ════════════════════════════════════════════════════════════════
# ROI 选取 + 两种信号模式
# ════════════════════════════════════════════════════════════════
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

def _roi_ts(frames, center):
    """提取单个ROI的预处理时序信号"""
    half = ROI_SIZE // 2
    r, c = center
    H, W = frames.shape[1], frames.shape[2]
    rs, re = max(0, r-half), min(H, r+half+1)
    cs, ce = max(0, c-half), min(W, c+half+1)
    ts = frames[:, rs:re, cs:ce].mean(axis=(1, 2))
    t  = np.arange(len(ts), dtype=np.float64)
    ts = ts - np.polyval(np.polyfit(t, ts, 3), t)  # poly_detrend
    ts = wavelet_denoise(ts, alpha=0.3)
    return ts.astype(np.float64)

def _ica_fuse(roi_mat, fs):
    from sklearn.decomposition import FastICA, PCA
    from scipy.fft import fft as _fft, fftfreq as _ff
    M, N = roi_mat.shape
    if M == 1: return roi_mat[0]
    X = roi_mat.T; nc = min(M, 5)
    try:
        src = FastICA(n_components=nc, random_state=42,
                      max_iter=2000, tol=1e-3).fit_transform(X)
    except Exception:
        src = PCA(n_components=nc, random_state=42).fit_transform(X)
    freqs = _ff(N, 1.0/fs)[:N//2]
    inb   = (freqs >= HB_LOW) & (freqs <= HB_HIGH)
    best, bsnr = None, -999.0
    for k in range(src.shape[1]):
        c   = src[:, k]
        psd = np.abs(_fft(c))[:N//2]**2
        snr = 10*np.log10(psd[inb].sum() / (psd[~inb&(freqs>0)].sum()+1e-9))
        if snr > bsnr: bsnr, best = snr, c
    ms = roi_mat.mean(axis=0)
    return best if np.dot(best, ms) >= 0 else -best

def _apply_vme(sig, fs, breath_freq):
    """单次VME去呼吸基线"""
    from algorithms.base import VME_Core
    try:
        u = VME_Core(sig - sig.mean(), fs=fs, f_init=breath_freq, alpha=1000)
        return sig - u
    except Exception:
        return sig

def build_signals(frames, breath_freq):
    """
    同时构建两种信号:
      sig_best1d: 能量最大的单ROI信号 (适合VMD)
      sig_fused:  ICA融合信号 (适合其他算法)
    两种信号都经过 VME 去呼吸基线 + 带通0.8-2.2Hz
    """
    mf = frames.mean(axis=0)
    sp = _split_col(mf)
    lc = _pick_centers(mf[:, :sp],  K_ROIS, MIN_DIST, 0)
    rc = _pick_centers(mf[:, sp:],  K_ROIS, MIN_DIST, sp)
    all_centers = lc + rc

    sigs = [_roi_ts(frames, c) for c in all_centers]
    for i, c in enumerate(all_centers):
        r, col = c
        print(f"       ROI{i+1}: 行{r:2d}列{col:2d}  压力={mf[r,col]:.1f}")

    roi_mat  = np.array(sigs)
    # 单最优ROI (能量最大)
    best_idx = int(np.argmax([np.std(s) for s in sigs]))
    sig_best = sigs[best_idx]
    # ICA融合
    sig_fuse = _ica_fuse(roi_mat, FS)

    # 两者都走 VME 去呼吸基线 + 带通
    sig_best = _apply_vme(sig_best, FS, breath_freq)
    sig_best = butter_bandpass_filter(sig_best, HB_LOW, HB_HIGH, fs=FS, order=4)
    sig_fuse = _apply_vme(sig_fuse, FS, breath_freq)
    sig_fuse = butter_bandpass_filter(sig_fuse, HB_LOW, HB_HIGH, fs=FS, order=4)

    print(f"       单ROI能量最大: ROI{best_idx+1}  "
          f"ICA融合 shape={sig_fuse.shape}")
    return sig_best, sig_fuse, mf


# ════════════════════════════════════════════════════════════════
# 算法执行（每种算法使用其最优输入+BPM策略）
# ════════════════════════════════════════════════════════════════
# 基于 v1~v4 实验结论的每算法最优策略表:
ALGO_STRATEGY = {
    '均值法':      ('fused', 'acr'),
    'ACMD':        ('fused', 'fpr'),
    'VMD':         ('best1d','fpr'),   # v1 单ROI+FPR最佳
    'EMD':         ('fused', 'fft'),   # v2 ICA+FFT最佳
    'VME':         ('fused', 'fpr'),
}


def run_all(sig_best, sig_fused, frames):
    algo_fn = {
        '均值法': lambda s, f: extract_heartbeat_mean(s, fs=FS),
        'ACMD':   lambda s, f: extract_heartbeat_acmd(s, fs=FS),
        'VMD':    lambda s, f: extract_heartbeat_vmd (s, fs=FS),
        'EMD':    lambda s, f: extract_heartbeat_emd (s, fs=FS),
        'VME':    lambda s, f: extract_heartbeat_vme (s, fs=FS),
    }
    bpm_fn = {'fpr': bpm_fpr, 'fft': bpm_fft, 'acr': bpm_acr}

    print('\n[ALGO]  (每算法独立最优策略)')
    results = {}
    for name in ALGO_NAMES:
        sig_mode, bpm_mode = ALGO_STRATEGY[name]
        sig_in = sig_best if sig_mode == 'best1d' else sig_fused
        print(f"    {name}[{sig_mode}+{bpm_mode}]...", end='', flush=True)
        t0 = time.perf_counter()
        try:
            out = np.array(algo_fn[name](sig_in, frames),
                           dtype=np.float64).flatten()
            if len(out) != len(sig_in):
                out = (out[:len(sig_in)] if len(out) > len(sig_in)
                       else np.pad(out, (0, len(sig_in)-len(out))))
            bpm = bpm_fn[bpm_mode](out, FS)
            # 若主估计失效，三方投票兜底
            if not (BPM_MIN <= bpm <= BPM_MAX):
                vals = [bpm_fpr(out,FS), bpm_fft(out,FS), bpm_acr(out,FS)]
                valid = [v for v in vals if BPM_MIN <= v <= BPM_MAX]
                bpm = float(np.median(valid)) if valid else 0.0
        except Exception as e:
            print(f" [err:{e}]")
            out = np.zeros_like(sig_in); bpm = 0.0
        el = (time.perf_counter()-t0)*1000
        results[name] = {'sig': out, 'bpm': float(bpm),
                         'mode': f"{sig_mode}+{bpm_mode}", 'time_ms': float(el)}
        print(f"  BPM={bpm:.1f}  耗时={el:.0f}ms")
    return results


# ════════════════════════════════════════════════════════════════
# 滑窗 + 单图 + 汇总图 + CSV + main()
# ════════════════════════════════════════════════════════════════
def sw_bpm_vote(sig, fs):
    win, step = int(WIN_SEC*fs), int(STEP_SEC*fs)
    T, B = [], []
    i = 0
    while i + win <= len(sig):
        seg = sig[i:i+win]
        vals = [bpm_fpr(seg,fs), bpm_fft(seg,fs), bpm_acr(seg,fs)]
        valid = [v for v in vals if BPM_MIN <= v <= BPM_MAX]
        T.append((i+win/2)/fs)
        B.append(float(np.median(valid)) if valid else 0.0)
        i += step
    return np.array(T), np.array(B)


def plot_one(name, res, ref_sig, ref_fs, ref_bpm, out_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7),
                                    constrained_layout=True)
    err = abs(res['bpm'] - ref_bpm)
    fig.suptitle(
        f'[{name}]  策略={res["mode"]}  BPM={res["bpm"]:.1f}  |  '
        f'参考={ref_bpm:.1f}  |  误差={err:.1f}  |  耗时={res["time_ms"]:.0f}ms',
        fontsize=11, fontweight='bold')

    rT, rB = sw_bpm_vote(ref_sig, ref_fs)
    aT, aB = sw_bpm_vote(res['sig'], FS)
    ax1.plot(rT, rB, 'r-',  lw=2.2, label=f'参考PPG ({ref_bpm:.1f})')
    ax1.plot(aT, aB, color='#27ae60', lw=1.8, label=f'{name} ({res["bpm"]:.1f})')
    ax1.axhline(ref_bpm, color='red', lw=0.8, ls='--', alpha=0.4)
    ax1.set_ylabel('BPM');  ax1.set_xlabel('时间(s)')
    ax1.set_title(f'滑动窗口BPM (三方投票, {WIN_SEC:.0f}s窗口)')
    ax1.legend(fontsize=10);  ax1.grid(alpha=0.25)
    yc = ref_bpm if ref_bpm > 0 else 75
    ax1.set_ylim(max(0, yc-30), yc+30)

    def _n(s): sd=np.std(s); return s/sd if sd>1e-9 else s
    ax2.plot(np.arange(len(ref_sig))/ref_fs, _n(ref_sig),
             'r-', lw=1.5, alpha=0.8, label='参考PPG')
    ax2.plot(np.arange(len(res['sig']))/FS, _n(res['sig']),
             color='#2980b9', lw=1.2, alpha=0.8, label=res['mode'])
    ax2.set_ylabel('归一化幅值');  ax2.set_xlabel('时间(s)')
    ax2.set_title('提取信号波形（标准差归一化）')
    ax2.legend(fontsize=9);  ax2.grid(alpha=0.25);  ax2.set_ylim(-5, 5)

    safe = name.replace(' ', '_').replace('-', '_')
    path = os.path.join(out_dir, f'{safe}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {name:<12} → {os.path.basename(path)}")


def plot_summary(all_res, ref_bpm, out_dir):
    names  = list(all_res.keys())
    errors = [abs(all_res[n]['bpm'] - ref_bpm) for n in names]
    times  = [all_res[n]['time_ms'] for n in names]
    modes  = [all_res[n]['mode']    for n in names]

    fig, ax1 = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax2 = ax1.twinx(); x, w = np.arange(len(names)), 0.35
    c_e = ['#27ae60' if e<=5 else '#f39c12' if e<=10 else '#e74c3c' for e in errors]
    b1 = ax1.bar(x-w/2, errors, w, color=c_e,     alpha=0.88, label='误差(左轴)')
    b2 = ax2.bar(x+w/2, times,  w, color='#5b9bd5',alpha=0.72, label='耗时ms(右轴)')
    for bar, v, m in zip(b1, errors, modes):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    for bar, v in zip(b2, times):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f'{v:.0f}', ha='center', va='bottom', fontsize=8, color='#2e6da4')
    ax1.axhline(5,  color='green',  lw=1.2, ls='--', alpha=0.7, label='±5 BPM')
    ax1.axhline(10, color='orange', lw=1.2, ls='--', alpha=0.7, label='±10 BPM')
    ax1.set_xticks(x); ax1.set_xticklabels(
        [f'{n}\n({all_res[n]["mode"]})' for n in names], fontsize=9)
    ax1.set_ylabel('|BPM误差|'); ax2.set_ylabel('耗时(ms)', color='#2e6da4')
    ax1.set_title(f'心跳算法最终版 (参考={ref_bpm:.1f} BPM, 每算法独立最优策略)')
    l1,lb1=ax1.get_legend_handles_labels(); l2,lb2=ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lb1+lb2, fontsize=8, loc='upper left')
    ax1.grid(alpha=0.25, axis='y')
    path = os.path.join(out_dir, '汇总误差与耗时.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  汇总图 → {path}")


def save_csv(all_res, ref_bpm, out_dir):
    # 按版本汇总对比
    VERSION_BEST = {
        'v1-VMD':  71.2,
        'v2-EMD':  80.2,
        'v3-均值法': 64.4,
        'v4-均值法': 65.4,
    }
    path = os.path.join(out_dir, '最终结果汇总.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['算法', '策略', '最优BPM', '参考BPM',
                    '绝对误差', '相对误差(%)', '耗时(ms)'])
        for nm, r in all_res.items():
            err = abs(r['bpm']-ref_bpm)
            rel = err/ref_bpm*100 if ref_bpm > 0 else float('nan')
            w.writerow([nm, r['mode'], f"{r['bpm']:.2f}",
                        f"{ref_bpm:.2f}", f"{err:.2f}",
                        f"{rel:.1f}", f"{r['time_ms']:.0f}"])
        w.writerow([])
        w.writerow(['===历史各版本最优==='])
        w.writerow(['版本', '算法', 'BPM', '参考', '误差'])
        for vn, b in VERSION_BEST.items():
            w.writerow([vn, '', f"{b:.1f}", f"{ref_bpm:.2f}",
                        f"{abs(b-ref_bpm):.2f}"])
    print(f"  CSV → {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print('\n'+'='*64)
    print('  心跳提取算法最终版 (每算法独立最优策略)')
    print(f'  输出: {OUT_DIR}')
    print('='*64)

    frames = load_cushion(DATA_FILE)
    ref_sig, ref_fs, ref_bpm, breath_freq = load_ref_ch2(REF_FILE)

    print('\n[预处理]')
    t0 = time.perf_counter()
    sig_best, sig_fused, mf = build_signals(frames, breath_freq)
    print(f"       耗时: {(time.perf_counter()-t0)*1000:.0f}ms")

    all_res = run_all(sig_best, sig_fused, frames)

    print('\n[PLOT]')
    for name in ALGO_NAMES:
        plot_one(name, all_res[name], ref_sig, ref_fs, ref_bpm, OUT_DIR)
    plot_summary(all_res, ref_bpm, OUT_DIR)

    print('\n[CSV]')
    save_csv(all_res, ref_bpm, OUT_DIR)

    print('\n'+'='*64)
    print(f'  参考={ref_bpm:.1f} BPM  |  呼吸={breath_freq*60:.1f}  '
          f'|  6次谐波={breath_freq*6*60:.1f} BPM')
    print(f'  {"算法":<12}  {"策略":<18}  {"BPM":>7}  {"误差":>7}  {"耗时ms":>8}')
    print(f'  {"-"*58}')
    for nm in ALGO_NAMES:
        r = all_res[nm]; err = abs(r['bpm']-ref_bpm)
        flg = 'OK' if err<=5 else ('~' if err<=10 else 'X')
        print(f'  {nm:<12}  {r["mode"]:<18}  {r["bpm"]:>7.1f}  '
              f'{err:>7.1f}  {r["time_ms"]:>8.0f}  {flg}')
    print('\n  历史最优: v1-VMD=71.2(err4.1)  v2-EMD=80.2(err4.8)')
    print('='*64+'\n')
    print(f'完成 → {OUT_DIR}\n')


if __name__ == '__main__':
    main()
