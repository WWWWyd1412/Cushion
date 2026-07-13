# -*- coding: utf-8 -*-
"""
心跳提取算法对比 heartbeat_contrast_corrected.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
正确流程（与 40_40_Extraction_1 原始架构一致）:

  frames (N,40,40)
    → get_dual_roi_mean_heartbeat(frames, fs)
        ① 定位左/右臀最大压力ROI中心(5×5)
        ② 提取两侧时序均值信号
        ③ VME_Core(f_init=0.25Hz) 剥除呼吸基线漂移
        ④ 小波去噪 (alpha=0.3)
        ⑤ 带通 0.8–2.2 Hz
    → sig_1d  (已含单VME基线去除)
    → 各算法: 均值法 / ACMD / VMD / EMD

  frames → extract_heartbeat_vme(frames, fs)
        (内部调用 get_dual_roi_mean_heartbeat + VME提取心跳)

BPM估计: ACR (自相关基础周期法) 为主，FPR为辅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys, os, csv, time, warnings
from typing import Any, cast
warnings.filterwarnings('ignore')

_DIR = os.path.dirname(os.path.abspath(__file__))
# 文件在 Contrast/心跳/ 下，需上溯两层到项目根目录 new_CUSHION/
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
    butter_bandpass_filter as _butter_bandpass_filter,
    wavelet_denoise as _wavelet_denoise,
    get_dual_roi_mean_heartbeat,
)
from algorithms.heartbeat_extract import (
    extract_heartbeat_mean, extract_heartbeat_acmd,
    extract_heartbeat_vmd,  extract_heartbeat_emd,
    extract_heartbeat_vme,
)


def butter_bandpass_filter(data: Any, lowcut: float = 0.1, highcut: float = 0.5,
                           fs: float = 10.0, order: int = 3) -> np.ndarray:
    """类型收窄包装：避免 Pylance 将外部函数推断为 None/tuple。"""
    return np.asarray(_butter_bandpass_filter(data, lowcut, highcut, fs, order), dtype=np.float64)


def wavelet_denoise(data: Any, alpha: float = 0.5) -> np.ndarray:
    """类型收窄包装：外部算法实际返回数组。"""
    return np.asarray(_wavelet_denoise(data, alpha), dtype=np.float64)

# ── 配置 ─────────────────────────────────────────────────────────
FS       = 11.2
TRIM_SEC = 20.0
DEADZONE = 30
CLIP_MAX = 2000
HB_LOW, HB_HIGH = 0.8, 2.2
BPM_MIN, BPM_MAX = 40.0, 150.0
WIN_SEC  = 25.0
STEP_SEC = 5.0

DATA_FILE = os.path.join(ROOT, 'data',         '20260702_160410_40x40.txt')
REF_FILE  = os.path.join(ROOT, 'Precise_Data', '刘若红0702.txt')
OUT_DIR   = os.path.join(ROOT, 'Contrast', '心跳', '刘若红_0702_160410_心跳_VME流程')

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
# BPM 估计器
# ════════════════════════════════════════════════════════════════
def bpm_acr(sig, fs):
    """自相关基础周期法 — 主估计器"""
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


def bpm_fpr(sig, fs):
    """FPR 峰值计数法 — 辅助估计器"""
    md = max(1, int(fs * 60 / BPM_MAX))
    pks, _ = find_peaks(sig, distance=md,
                        prominence=np.ptp(sig)*0.1 if np.ptp(sig) > 0 else 0.01)
    if len(pks) < 2: return 0.0
    b = 60 * fs / np.mean(np.diff(pks))
    return float(b) if BPM_MIN <= b <= BPM_MAX else 0.0


def bpm_fft(sig, fs):
    """FFT 主频法"""
    n = len(sig)
    freqs = np.fft.rfftfreq(n, 1.0/fs)
    psd   = np.abs(np.fft.rfft(sig - sig.mean()))**2
    mask  = (freqs >= BPM_MIN/60) & (freqs <= BPM_MAX/60)
    if not mask.any(): return 0.0
    return float(freqs[mask][np.argmax(psd[mask])] * 60)


def best_bpm(sig, fs):
    """
    ACR 优先策略:
      ACR 有效 → 直接使用 ACR
      ACR 失效 → 降级到 FPR
      FPR 也失效 → FFT 兜底
    心跳信号中 ACR 最不受谐波影响，是最可靠的估计器。
    """
    a = bpm_acr(sig, fs)
    p = bpm_fpr(sig, fs)
    f = bpm_fft(sig, fs)
    if BPM_MIN <= a <= BPM_MAX:
        vote = a
    elif BPM_MIN <= p <= BPM_MAX:
        vote = p
    elif BPM_MIN <= f <= BPM_MAX:
        vote = f
    else:
        vote = 0.0
    return float(vote), a, p, f


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
    if len(frames) > 2*trim:
        frames = frames[trim:-trim]
    print(f"       {len(frames)} 帧 ({len(frames)/FS:.1f}s)")
    return frames


def load_ref_ch2(fp):
    """解析 CH2 (PPG) 作为心率参考，同时读 CH1 (RSP) 用于标注呼吸频率"""
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
        try:
            ch1r.append(float(cols[0]))
            ch2r.append(float(cols[1]))
        except: continue

    trim = int(TRIM_SEC * fs_r)

    # CH2 → 参考心率
    ch2 = np.array(ch2r, dtype=np.float64)
    if len(ch2) > 2*trim: ch2 = ch2[trim:-trim]
    ch2 -= ch2.mean()
    b2, a2 = cast(tuple[np.ndarray, np.ndarray], butter(4, 5.0/(0.5*fs_r), btype='low', output='ba'))
    ds     = max(1, int(fs_r/50))
    ch2_ds = filtfilt(b2, a2, ch2)[::ds]
    fs_ppg = fs_r / ds
    ch2_bp = butter_bandpass_filter(ch2_ds, HB_LOW, HB_HIGH, fs=fs_ppg, order=4)
    vote, a, p, f = best_bpm(ch2_bp, fs_ppg)
    print(f"       CH2(PPG)参考心率: ACR={a:.1f} FPR={p:.1f} FFT={f:.1f} → 中位={vote:.1f} BPM")
    return ch2_bp, float(fs_ppg), float(vote)


# ════════════════════════════════════════════════════════════════
# 多ROI + ICA + VME 预处理（与呼吸流程统一，频段改为0.8–2.2Hz）
# ════════════════════════════════════════════════════════════════
def _split_col(mf):
    """自适应左右分割列：按列压力总和找谷值，约束在12–28列"""
    return 12 + int(np.argmin(mf.sum(axis=0)[12:28]))


def _pick_centers(zone, k, min_d, c_off):
    """在zone区域按压力降序选k个ROI中心，最小间距min_d"""
    order = np.argsort(zone.ravel())[::-1]
    cens  = []
    for idx in order:
        r, cl = np.unravel_index(idx, zone.shape)
        c = cl + c_off
        if not any(max(abs(r-cr), abs(c-cc)) < min_d for cr,cc in cens):
            cens.append((r, c))
        if len(cens) == k: break
    while len(cens) < k:
        cens.append((zone.shape[0]//2, c_off + zone.shape[1]//2))
    return cens


def _ica_fuse_hb(roi_mat, fs):
    """ICA融合：选心跳频段[0.8–2.2Hz] SNR最高的独立分量"""
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
    best, bsnr = src[:, 0], -999.0
    for k in range(src.shape[1]):
        c   = src[:, k]
        psd = np.abs(_fft(c))[:N//2]**2
        snr = 10*np.log10(psd[inb].sum() /
                          (psd[~inb & (freqs>0)].sum() + 1e-9))
        if snr > bsnr: bsnr, best = snr, c
    ms = roi_mat.mean(axis=0)
    return best if np.dot(best, ms) >= 0 else -best


def get_standard_hb_signal(frames):
    """
    统一多ROI + ICA + VME 心跳预处理流程:
      ① 自适应左右分割列 (12–28列间的压力谷值)
      ② 左/右各选4个ROI中心 (3×3, 最小间距5px, 按压力降序)
      ③ 每个ROI: 3×3均值时序 → 多项式去趋势(order=3)
                → 小波去噪(alpha=0.3) → 带通0.8–2.2Hz
      ④ FastICA融合 → 心跳频段SNR最高的独立分量
      ⑤ VME_Core(f_init=0.25Hz) 剥除呼吸基线漂移
      ⑥ 带通0.8–2.2Hz (VME后再滤一次保证频段干净)
    """
    from algorithms.base import VME_Core
    print("[PRE]  多ROI+ICA+VME 心跳预处理 ...")
    t0 = time.perf_counter()

    mf   = frames.mean(axis=0)
    sp   = _split_col(mf)
    lc   = _pick_centers(mf[:, :sp], 4, 5, 0)
    rc   = _pick_centers(mf[:, sp:], 4, 5, sp)
    all_centers = lc + rc

    half = 1  # ROI_SIZE=3 → half=1
    H, W = frames.shape[1], frames.shape[2]
    sigs = []
    for r, c in all_centers:
        rs, re = max(0, r-half), min(H, r+half+1)
        cs, ce = max(0, c-half), min(W, c+half+1)
        ts = frames[:, rs:re, cs:ce].mean(axis=(1, 2))
        # 多项式去趋势
        t2 = np.arange(len(ts), dtype=np.float64)
        ts = ts - np.polyval(np.polyfit(t2, ts, 3), t2)
        # 小波去噪 (alpha=0.3，比呼吸的0.5更小，保留心跳高频细节)
        ts = wavelet_denoise(ts, alpha=0.3)
        # 带通 0.8–2.2Hz
        ts = butter_bandpass_filter(ts, HB_LOW, HB_HIGH, fs=FS, order=3)
        sigs.append(ts.astype(np.float64))

    roi_mat = np.array(sigs)
    fused   = _ica_fuse_hb(roi_mat, FS)

    # VME 去呼吸基线
    try:
        # 先估算信号中的心跳主频，用于自适应alpha
        n = len(fused)
        freqs_est = np.fft.rfftfreq(n, 1.0/FS)
        psd_est   = np.abs(np.fft.rfft(fused - fused.mean()))**2
        vm = (freqs_est >= HB_LOW) & (freqs_est <= HB_HIGH)
        f_heart = float(freqs_est[vm][np.argmax(psd_est[vm])]) if vm.any() else 1.2
        alpha_bd = (1000.0 * np.exp(1.09 * ((f_heart - 1.25) / -0.5)**2)
                    if f_heart <= 1.25 else 1000.0)
        u_bd  = np.asarray(VME_Core(fused - fused.mean(), fs=FS, f_init=0.25, alpha=int(alpha_bd)), dtype=np.float64)
        fused = fused - u_bd
    except Exception:
        pass

    fused = butter_bandpass_filter(fused, HB_LOW, HB_HIGH, fs=FS, order=4)

    pre_ms = (time.perf_counter() - t0) * 1000
    print(f"       分割列={sp}  ROI数={len(all_centers)}  "
          f"shape={fused.shape}  耗时={pre_ms:.0f}ms")
    return fused


# ════════════════════════════════════════════════════════════════
# 算法执行
# ════════════════════════════════════════════════════════════════
def run_all(sig_1d: np.ndarray, frames: np.ndarray) -> dict:
    """
    sig_1d : 经 get_dual_roi_mean_heartbeat 预处理的1D信号
    frames : 原始帧序列（供 VME 直接调用3D分支）

    算法输入:
      均值法/ACMD/VMD/EMD → sig_1d (已含VME基线去除)
      VME                 → frames 直接输入（内部自行调用流程）
    BPM: ACR 为主，中位数兜底
    """
    algo_map = {
        '均值法': lambda: extract_heartbeat_mean(sig_1d, fs=FS),
        'ACMD':   lambda: extract_heartbeat_acmd(sig_1d, fs=FS),
        'VMD':    lambda: extract_heartbeat_vmd (sig_1d, fs=FS),
        'EMD':    lambda: extract_heartbeat_emd (sig_1d, fs=FS),
        'VME':    lambda: extract_heartbeat_vme (frames, fs=FS),  # 3D路径
    }

    print('\n[ALGO]')
    results = {}
    for name in ALGO_NAMES:
        print(f"    {name}...", end='', flush=True)
        t0 = time.perf_counter()
        try:
            out = np.array(algo_map[name](), dtype=np.float64).flatten()
            ref_len = len(sig_1d)
            if len(out) != ref_len:
                out = (out[:ref_len] if len(out) > ref_len
                       else np.pad(out, (0, ref_len - len(out))))
            vote, a, p, f = best_bpm(out, FS)
        except Exception as e:
            print(f" [err:{e}]")
            out = np.zeros_like(sig_1d); vote = a = p = f = 0.0
        el = (time.perf_counter() - t0) * 1000
        results[name] = {
            'sig': out, 'bpm': float(vote),
            'acr': float(a), 'fpr': float(p), 'fft': float(f),
            'time_ms': float(el),
        }
        print(f"  ACR={a:.1f} FPR={p:.1f} FFT={f:.1f} 投票={vote:.1f}  耗时={el:.0f}ms")
    return results


# ════════════════════════════════════════════════════════════════
# 滑动窗口 + 单算法图 + 汇总图 + CSV + main()
# ════════════════════════════════════════════════════════════════
def sw_bpm(sig, fs):
    win, step = int(WIN_SEC*fs), int(STEP_SEC*fs)
    T, B = [], []
    i = 0
    while i + win <= len(sig):
        b, _, _, _ = best_bpm(sig[i:i+win], fs)
        T.append((i + win/2) / fs)
        B.append(b)
        i += step
    return np.array(T), np.array(B)


def plot_one(name, res, ref_sig, ref_fs, ref_bpm, out_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7),
                                    constrained_layout=True)
    err = abs(res['bpm'] - ref_bpm)
    fig.suptitle(
        f'[{name}]  BPM={res["bpm"]:.1f}  |  参考={ref_bpm:.1f}  |  '
        f'误差={err:.1f} BPM  |  耗时={res["time_ms"]:.0f}ms\n'
        f'预处理: get_dual_roi_mean_heartbeat (单VME剥除呼吸基线)',
        fontsize=10, fontweight='bold')

    rT, rB = sw_bpm(ref_sig, ref_fs)
    aT, aB = sw_bpm(res['sig'], FS)
    ax1.plot(rT, rB, 'r-',  lw=2.2, label=f'参考PPG ({ref_bpm:.1f} BPM)')
    ax1.plot(aT, aB, color='#27ae60', lw=1.8,
             label=f'{name} ({res["bpm"]:.1f} BPM)')
    ax1.axhline(ref_bpm, color='red', lw=0.8, ls='--', alpha=0.4)
    ax1.set_ylabel('BPM');  ax1.set_xlabel('时间 (s)')
    ax1.set_title(f'滑动窗口BPM ({WIN_SEC:.0f}s / {STEP_SEC:.0f}s，三方投票)')
    ax1.legend(fontsize=10);  ax1.grid(alpha=0.25)
    yc = ref_bpm if ref_bpm > 0 else 75
    ax1.set_ylim(max(0, yc - 30), yc + 30)

    def _n(s): sd=np.std(s); return s/sd if sd > 1e-9 else s
    ax2.plot(np.arange(len(ref_sig))/ref_fs, _n(ref_sig),
             'r-', lw=1.5, alpha=0.8, label='参考PPG')
    ax2.plot(np.arange(len(res['sig']))/FS, _n(res['sig']),
             color='#2980b9', lw=1.2, alpha=0.8,
             label=f'单VME基线去除 + {name}')
    ax2.set_ylabel('归一化幅值');  ax2.set_xlabel('时间 (s)')
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

    fig, ax1 = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax2 = ax1.twinx()
    x, w = np.arange(len(names)), 0.35
    c_e = ['#27ae60' if e<=5 else '#f39c12' if e<=10 else '#e74c3c'
            for e in errors]
    b1 = ax1.bar(x-w/2, errors, w, color=c_e,      alpha=0.88, label='误差 (左轴)')
    b2 = ax2.bar(x+w/2, times,  w, color='#5b9bd5', alpha=0.72, label='耗时ms (右轴)')
    for bar, v in zip(b1, errors):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    for bar, v in zip(b2, times):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f'{v:.0f}', ha='center', va='bottom', fontsize=8, color='#2e6da4')
    ax1.axhline(5,  color='green',  lw=1.2, ls='--', alpha=0.7, label='±5 BPM')
    ax1.axhline(10, color='orange', lw=1.2, ls='--', alpha=0.7, label='±10 BPM')
    ax1.set_xticks(x);  ax1.set_xticklabels(names, fontsize=11)
    ax1.set_ylabel('|BPM误差|');  ax2.set_ylabel('耗时(ms)', color='#2e6da4')
    ax1.set_title(f'心跳算法对比 (参考={ref_bpm:.1f} BPM | 预处理=单VME基线去除)')
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lb1+lb2, fontsize=8, loc='upper left')
    ax1.grid(alpha=0.25, axis='y')
    path = os.path.join(out_dir, '汇总误差与耗时.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  汇总图 → {path}")


def save_csv(all_res, ref_bpm, out_dir):
    path = os.path.join(out_dir, '心跳结果汇总.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['算法', '投票BPM', 'ACR', 'FPR', 'FFT',
                    '参考BPM', '绝对误差', '相对误差(%)', '耗时(ms)',
                    '预处理'])
        for nm, r in all_res.items():
            err = abs(r['bpm'] - ref_bpm)
            rel = err/ref_bpm*100 if ref_bpm > 0 else float('nan')
            pre = ('frames→VME内部' if nm == 'VME'
                   else 'get_dual_roi_mean_heartbeat(VME)')
            w.writerow([nm,
                f"{r['bpm']:.2f}", f"{r['acr']:.2f}",
                f"{r['fpr']:.2f}", f"{r['fft']:.2f}",
                f"{ref_bpm:.2f}", f"{err:.2f}",
                f"{rel:.1f}", f"{r['time_ms']:.0f}", pre])
    print(f"  CSV → {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print('\n' + '='*64)
    print('  心跳提取对比  (标准VME单次基线去除流程)')
    print(f'  输出: {OUT_DIR}')
    print('='*64)

    frames = load_cushion(DATA_FILE)
    ref_sig, ref_fs, ref_bpm = load_ref_ch2(REF_FILE)

    # 标准VME预处理 —— 所有1D算法共用
    sig_1d = get_standard_hb_signal(frames)

    all_res = run_all(sig_1d, frames)

    print('\n[PLOT]')
    for name in ALGO_NAMES:
        plot_one(name, all_res[name], ref_sig, ref_fs, ref_bpm, OUT_DIR)
    plot_summary(all_res, ref_bpm, OUT_DIR)

    print('\n[CSV]')
    save_csv(all_res, ref_bpm, OUT_DIR)

    print('\n' + '='*64)
    print(f'  参考心率 (CH2 PPG) = {ref_bpm:.1f} BPM')
    print(f'  {"算法":<12}  {"投票BPM":>8}  {"ACR":>7}  {"FPR":>7}  '
          f'{"误差":>7}  {"耗时ms":>8}')
    print(f'  {"-"*58}')
    for nm in ALGO_NAMES:
        r   = all_res[nm]
        err = abs(r['bpm'] - ref_bpm)
        flg = 'OK' if err <= 5 else ('~' if err <= 10 else 'X')
        print(f'  {nm:<12}  {r["bpm"]:>8.1f}  {r["acr"]:>7.1f}  '
              f'{r["fpr"]:>7.1f}  {err:>7.1f}  {r["time_ms"]:>8.0f}  {flg}')
    print('='*64 + '\n')
    print(f'完成 → {OUT_DIR}\n')


if __name__ == '__main__':
    main()