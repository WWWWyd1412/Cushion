# -*- coding: utf-8 -*-
"""
呼吸算法最终对比版 v6.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
对全部10种算法应用各自最优策略:
  [所有算法] ICA多ROI融合信号作为输入
  [BPM估计 ] AutoCorr-BPM (自相关基础周期)
  [AFD     ] FFT主频初始化修复版
  [去趋势  ] 多项式去趋势 (order=3) 在ICA融合前执行

新增: 每种算法记录处理时间 (ms)

输出:
  Contrast/刘若红_0702_160410_v6最终/
    ├── {算法名}.png       每算法独立图 (滑窗BPM + 信号波形)
    ├── 汇总误差对比.png
    └── 最终结果汇总.csv
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
    calculate_bpm_fpr, butter_bandpass_filter, wavelet_denoise,
)
from algorithms.breath_extract import (
    extract_breath_mean, extract_breath_acmd,
    extract_breath_vmd,  extract_breath_emd,
    extract_breath_vmd_mape, extract_breath_goa_vmd,
    extract_breath_smvmd, extract_breath_mvmd,
    extract_breath_multi_roi_ica,
)

# ── 配置 ─────────────────────────────────────────────────────────
FS       = 11.2
TRIM_SEC = 20.0
ROI_SIZE = 3
K_ROIS   = 4
MIN_DIST = 5
WIN_SEC  = 30.0
STEP_SEC = 5.0
DEADZONE = 30
CLIP_MAX = 2000

DATA_FILE = os.path.join(ROOT, 'data',         '20260702_160410_40x40.txt')
REF_FILE  = os.path.join(ROOT, 'Precise_Data', '刘若红0702.txt')
OUT_DIR   = os.path.join(ROOT, 'Contrast', '呼吸', '刘若红_0702_160410_v6最终')

ALGO_NAMES = [
    '均值法', 'ACMD', 'VMD', 'EMD', 'AFD',
    'VMD-MAPE', 'GOA-VMD', 'SMVMD', 'MVMD', 'Multi-ROI ICA',
]

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
# BPM 估计: AutoCorr-BPM (所有算法统一使用)
# ════════════════════════════════════════════════════════════════
def bpm_acr(sig: np.ndarray, fs: float,
            min_bpm: float = 6.0, max_bpm: float = 40.0) -> float:
    """
    自相关基础周期BPM:
      1. 计算归一化自相关函数
      2. 在生理范围 [min_bpm, max_bpm] 对应的延迟区间内
         找第一个显著峰(prominence≥0.08)
      3. BPM = 60 / (峰延迟 / fs)
    优势: 直接找基础周期, 不受波形吸/呼双峰影响。
    """
    n = len(sig)
    if n < 20:
        return 0.0
    s   = sig - sig.mean()
    acf = correlate(s, s, mode='full')[n-1:]
    acf = acf / (acf[0] + 1e-12)

    lg_min = max(1, int(60.0 / max_bpm * fs))
    lg_max = min(n-1, int(60.0 / min_bpm * fs))
    if lg_min >= lg_max:
        return 0.0

    seg  = acf[lg_min:lg_max]
    pks, pr = find_peaks(seg, prominence=0.08)
    pk = (lg_min + int(pks[np.argmax(pr['prominences'])])
          if len(pks) else lg_min + int(np.argmax(seg)))

    bpm = 60.0 / (pk / fs)
    return float(bpm) if min_bpm <= bpm <= max_bpm else 0.0


# ════════════════════════════════════════════════════════════════
# 信号预处理工具
# ════════════════════════════════════════════════════════════════
def poly_detrend(sig: np.ndarray, order: int = 3) -> np.ndarray:
    """多项式去趋势 — 去除传感器漂移引起的低频缓慢变化"""
    t      = np.arange(len(sig), dtype=np.float64)
    coeffs = np.polyfit(t, sig, order)
    return sig - np.polyval(coeffs, t)


def preprocess_roi_signal(ts: np.ndarray) -> np.ndarray:
    """单ROI时序预处理: 多项式去趋势 → 小波去噪 → 带通0.1–0.5Hz"""
    ts = poly_detrend(ts, order=3)
    ts = wavelet_denoise(ts, alpha=0.5)
    ts = butter_bandpass_filter(ts, 0.1, 0.5, fs=FS, order=3)
    return ts.astype(np.float64)


# ════════════════════════════════════════════════════════════════
# AFD 修复版 — FFT主频初始化
# ════════════════════════════════════════════════════════════════
def extract_breath_afd_fixed(signal: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    修复版AFD:
      1. FFT在0.08–0.6Hz找主频 f_coarse
      2. 在 f_coarse±0.05Hz 范围内100点细化搜索
      3. 以最优频率做余弦/正弦单模态投影
    """
    n = len(signal)
    if n < 30:
        return signal.copy()
    sig = signal - signal.mean()

    freqs = np.fft.rfftfreq(n, 1.0/fs)
    psd   = np.abs(np.fft.rfft(sig))**2
    mask  = (freqs >= 0.08) & (freqs <= 0.6)
    if not mask.any():
        return sig
    f0 = freqs[mask][np.argmax(psd[mask])]

    f_lo, f_hi = max(0.08, f0 - 0.05), min(0.60, f0 + 0.05)
    t = np.arange(n) / fs
    best_f, best_e = f0, -1.0
    for f in np.linspace(f_lo, f_hi, 100):
        c = np.cos(2*np.pi*f*t);  s = np.sin(2*np.pi*f*t)
        e = (np.dot(sig,c)**2 + np.dot(sig,s)**2) / n
        if e > best_e:
            best_e, best_f = e, f

    c = np.cos(2*np.pi*best_f*t);  s = np.sin(2*np.pi*best_f*t)
    return (c * np.dot(sig,c)/(np.dot(c,c)+1e-9) +
            s * np.dot(sig,s)/(np.dot(s,s)+1e-9))


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


def load_ref(fp):
    print(f"[REF]  {os.path.basename(fp)}")
    fs_r = 2000.0
    with open(fp, 'r', encoding='utf-8', errors='replace') as fh:
        lines = fh.readlines()
    for ln in lines[:4]:
        if 'msec/sample' in ln:
            try: fs_r = 1000.0 / float(ln.strip().split()[0])
            except: pass
            break
    di = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith('CH1'): di = i+2; break
    raw = []
    for ln in lines[di:]:
        try: raw.append(float(ln.strip().split('\t')[0]))
        except: continue
    raw = np.array(raw, dtype=np.float64)
    trim = int(TRIM_SEC * fs_r)
    if len(raw) > 2*trim: raw = raw[trim:-trim]
    raw -= raw.mean()
    b, a  = butter(4, 1.0/(0.5*fs_r), btype='low')
    lp    = filtfilt(b, a, raw)
    ds    = max(1, int(fs_r/10.0))
    sig   = butter_bandpass_filter(lp[::ds], 0.1, 0.5, fs=fs_r/ds, order=4)
    ref_bpm = bpm_acr(sig, fs=fs_r/ds)
    print(f"       参考BPM (ACR) = {ref_bpm:.2f}")
    return sig, float(fs_r/ds), float(ref_bpm)


# ════════════════════════════════════════════════════════════════
# ROI选取 + 多项式去趋势 + ICA融合
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


def build_fused_signal(frames):
    """
    选取左/右两侧各K_ROIS个ROI中心(3×3窗口),
    对每个ROI做 poly_detrend+小波+带通预处理,
    最后用FastICA在呼吸频段SNR最高的独立分量作为融合信号。
    返回: fused (N,), rois列表, mean_frame
    """
    mf   = frames.mean(axis=0)
    sp   = _split_col(mf)
    lc   = _pick_centers(mf[:, :sp], K_ROIS, MIN_DIST, 0)
    rc   = _pick_centers(mf[:, sp:], K_ROIS, MIN_DIST, sp)
    rois = ([{'label':f'L{i+1}','c':c} for i,c in enumerate(lc)] +
            [{'label':f'R{i+1}','c':c} for i,c in enumerate(rc)])

    half = ROI_SIZE // 2
    H, W = frames.shape[1], frames.shape[2]
    sigs = []
    for roi in rois:
        r, c = roi['c']
        rs, re = max(0,r-half), min(H,r+half+1)
        cs, ce = max(0,c-half), min(W,c+half+1)
        ts = frames[:, rs:re, cs:ce].mean(axis=(1,2))
        sigs.append(preprocess_roi_signal(ts))
        print(f"       {roi['label']}: 行{r:2d}列{c:2d}  压力={mf[r,c]:.1f}")

    roi_mat = np.array(sigs)   # (M, N)
    fused   = _ica_fuse(roi_mat, FS)
    print(f"       ICA融合完成, shape={fused.shape}")
    return fused, rois, mf


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
    inb   = (freqs >= 0.1) & (freqs <= 0.5)
    best, bsnr = None, -999.0
    for k in range(src.shape[1]):
        c   = src[:, k]
        psd = np.abs(_fft(c))[:N//2]**2
        snr = 10*np.log10(psd[inb].sum() /
                          (psd[~inb & (freqs>0)].sum() + 1e-9))
        if snr > bsnr: bsnr, best = snr, c
    ms = roi_mat.mean(axis=0)
    return best if np.dot(best, ms) >= 0 else -best


# ════════════════════════════════════════════════════════════════
# 算法执行：最优策略 + 计时
# ════════════════════════════════════════════════════════════════
def run_all(fused: np.ndarray, frames: np.ndarray) -> dict:
    """
    每种算法:
      输入  = ICA融合1D信号 (fused)
      BPM   = AutoCorr-BPM
      计时  = 算法提取 + BPM估计的总耗时 (ms)
    特殊: AFD 使用 FFT主频初始化修复版
    """
    algo_map = {
        '均值法':        lambda s, f: extract_breath_mean(s),
        'ACMD':           lambda s, f: extract_breath_acmd(s,  fs=FS),
        'VMD':            lambda s, f: extract_breath_vmd (s,  fs=FS),
        'EMD':            lambda s, f: extract_breath_emd (s,  fs=FS),
        'AFD':            lambda s, f: extract_breath_afd_fixed(s, fs=FS),
        'VMD-MAPE':       lambda s, f: extract_breath_vmd_mape(s, fs=FS),
        'GOA-VMD':        lambda s, f: extract_breath_goa_vmd (s, fs=FS),
        'SMVMD':          lambda s, f: extract_breath_smvmd(s,   fs=FS),
        'MVMD':           lambda s, f: extract_breath_mvmd (s,   fs=FS),
        'Multi-ROI ICA':  lambda s, f: extract_breath_multi_roi_ica(s, fs=FS),
    }

    print('\n[ALGO]  (输入=ICA融合信号, BPM=AutoCorr)')
    results = {}
    for name in ALGO_NAMES:
        print(f"    {name}...", end='', flush=True)
        t0 = time.perf_counter()
        try:
            sig = algo_map[name](fused, frames)
            bpm = bpm_acr(sig, fs=FS)
        except Exception as e:
            print(f" [err:{e}]")
            sig = np.zeros_like(fused)
            bpm = 0.0
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results[name] = {
            'sig':     sig,
            'bpm':     float(bpm),
            'time_ms': float(elapsed_ms),
        }
        print(f"  BPM={bpm:.2f}  耗时={elapsed_ms:.1f}ms")

    return results


# ════════════════════════════════════════════════════════════════
# 滑动窗口 ACR-BPM 曲线
# ════════════════════════════════════════════════════════════════
def sw_acr(sig, fs):
    win, step = int(WIN_SEC*fs), int(STEP_SEC*fs)
    T, B = [], []
    i = 0
    while i + win <= len(sig):
        T.append((i + win/2) / fs)
        B.append(bpm_acr(sig[i:i+win], fs))
        i += step
    return np.array(T), np.array(B)


# ════════════════════════════════════════════════════════════════
# 单算法独立图（2面板）
# ════════════════════════════════════════════════════════════════
def plot_one(name, res, ref_sig, ref_fs, ref_bpm, out_dir):
    """
    面板1 (上): 滑动窗口ACR-BPM曲线
      红线  = 参考RSP
      绿线  = 算法提取 (ACR BPM)
      红虚线 = 参考全段BPM水平线

    面板2 (下): 提取信号波形 (标准差归一化)
      红   = 参考RSP
      蓝   = ICA+算法输出
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7),
                                    constrained_layout=True)
    err = abs(res['bpm'] - ref_bpm)
    fig.suptitle(
        f'[{name}]   BPM={res["bpm"]:.2f}  |  '
        f'参考={ref_bpm:.2f}  |  误差={err:.2f} BPM  |  '
        f'耗时={res["time_ms"]:.1f} ms',
        fontsize=11, fontweight='bold'
    )

    # ── 面板1: 滑窗BPM ──
    rT, rB = sw_acr(ref_sig, ref_fs)
    aT, aB = sw_acr(res['sig'], FS)

    ax1.plot(rT, rB, 'r-',  lw=2.2,  label=f'参考RSP  (ACR={ref_bpm:.2f})')
    ax1.plot(aT, aB, color='#27ae60', lw=1.8,
             label=f'{name} (ACR={res["bpm"]:.2f})')
    ax1.axhline(ref_bpm, color='red', lw=0.9, ls='--', alpha=0.5)

    ax1.set_ylabel('BPM');  ax1.set_xlabel('时间 (s)')
    ax1.set_title(f'滑动窗口ACR-BPM  (窗口={WIN_SEC:.0f}s / 步长={STEP_SEC:.0f}s)')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(alpha=0.25)
    ax1.set_ylim(max(0, ref_bpm - 12), ref_bpm + 12)

    # ── 面板2: 信号波形 ──
    def _n(s): sd = np.std(s); return s/sd if sd > 1e-9 else s
    t_r = np.arange(len(ref_sig)) / ref_fs
    t_a = np.arange(len(res['sig'])) / FS

    ax2.plot(t_r, _n(ref_sig),    'r-',            lw=1.6, alpha=0.85, label='参考RSP')
    ax2.plot(t_a, _n(res['sig']), color='#2980b9', lw=1.3, alpha=0.80,
             label=f'ICA+{name}')
    ax2.set_ylabel('归一化幅值');  ax2.set_xlabel('时间 (s)')
    ax2.set_title('提取信号波形（标准差归一化）')
    ax2.legend(fontsize=9);  ax2.grid(alpha=0.25);  ax2.set_ylim(-5, 5)

    safe = name.replace(' ','_').replace('-','_').replace('/','_')
    path = os.path.join(out_dir, f'{safe}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {name:<18} → {os.path.basename(path)}")


# ════════════════════════════════════════════════════════════════
# 汇总图：误差 + 处理时间双坐标
# ════════════════════════════════════════════════════════════════
def plot_summary(all_res, ref_bpm, out_dir):
    names  = list(all_res.keys())
    errors = [abs(all_res[n]['bpm'] - ref_bpm)  for n in names]
    times  = [all_res[n]['time_ms']              for n in names]

    fig, ax1 = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax2 = ax1.twinx()

    x  = np.arange(len(names))
    w  = 0.38
    c_err  = ['#27ae60' if e<=1.5 else '#f39c12' if e<=3 else '#e74c3c'
               for e in errors]
    bars1 = ax1.bar(x - w/2, errors, w, color=c_err, alpha=0.88,
                    label='BPM误差 (左轴)')
    bars2 = ax2.bar(x + w/2, times,  w, color='#5b9bd5', alpha=0.72,
                    label='处理时间 ms (右轴)')

    for bar, v in zip(bars1, errors):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.08,
                 f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    for bar, v in zip(bars2, times):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                 f'{v:.0f}', ha='center', va='bottom', fontsize=8,
                 color='#2e6da4')

    ax1.axhline(1.5, color='green',  lw=1.2, ls='--', alpha=0.7,
                label='±1.5 BPM阈值')
    ax1.axhline(3.0, color='orange', lw=1.2, ls='--', alpha=0.7,
                label='±3.0 BPM阈值')
    ax1.set_xticks(x);  ax1.set_xticklabels(names, rotation=25, ha='right')
    ax1.set_ylabel('|BPM误差|', color='#333')
    ax2.set_ylabel('处理时间 (ms)', color='#2e6da4')
    ax1.set_title(f'各算法BPM误差 + 处理时间  (参考={ref_bpm:.2f} BPM, ICA+ACR最优策略)',
                  fontsize=11)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, fontsize=8, loc='upper left')

    path = os.path.join(out_dir, '汇总误差与耗时.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  汇总图 → {path}")


# ════════════════════════════════════════════════════════════════
# CSV（含处理时间）
# ════════════════════════════════════════════════════════════════
def save_csv(all_res, ref_bpm, out_dir):
    path = os.path.join(out_dir, '最终结果汇总.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['算法', '提取BPM(ACR)', '参考BPM(ACR)',
                    '绝对误差', '相对误差(%)', '处理时间(ms)', '策略'])
        for nm, r in all_res.items():
            err = abs(r['bpm'] - ref_bpm)
            rel = err / ref_bpm * 100 if ref_bpm > 0 else float('nan')
            strategy = 'ICA融合 + FFT主频初始化 + ACR' if nm == 'AFD' \
                       else 'ICA融合 + ACR'
            w.writerow([nm,
                        f"{r['bpm']:.3f}", f"{ref_bpm:.3f}",
                        f"{err:.3f}", f"{rel:.2f}",
                        f"{r['time_ms']:.1f}", strategy])
    print(f"  CSV → {path}")


# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print('\n' + '='*64)
    print('  呼吸算法最终对比版 v6.0  (ICA+ACR最优策略 + 处理计时)')
    print(f'  输出: {OUT_DIR}')
    print('='*64)

    # 1. 数据加载
    frames = load_cushion(DATA_FILE)
    ref_sig, ref_fs, ref_bpm = load_ref(REF_FILE)

    # 2. ROI + ICA融合（ICA本身的时间不算入各算法）
    print('\n[ROI]')
    t_pre = time.perf_counter()
    fused, rois, mf = build_fused_signal(frames)
    pre_ms = (time.perf_counter() - t_pre) * 1000
    print(f"       预处理(ROI+ICA)耗时: {pre_ms:.1f}ms")

    # 3. 所有算法
    all_res = run_all(fused, frames)

    # 4. 单算法图
    print('\n[PLOT] 单算法图:')
    for name in ALGO_NAMES:
        plot_one(name, all_res[name], ref_sig, ref_fs, ref_bpm, OUT_DIR)

    # 5. 汇总图
    print('\n[PLOT] 汇总图:')
    plot_summary(all_res, ref_bpm, OUT_DIR)

    # 6. CSV
    print('\n[CSV]')
    save_csv(all_res, ref_bpm, OUT_DIR)

    # 7. 控制台汇总
    print('\n' + '='*64)
    print(f'  参考BPM (ACR) = {ref_bpm:.2f}')
    print(f'  {"算法":<18}  {"BPM":>7}  {"误差":>7}  {"耗时(ms)":>10}  {"策略"}')
    print(f'  {"-"*60}')
    for nm in ALGO_NAMES:
        r   = all_res[nm]
        err = abs(r['bpm'] - ref_bpm)
        flg = 'OK' if err<=1.5 else ('~' if err<=3 else 'X')
        strat = '(FFT初始化)' if nm == 'AFD' else ''
        print(f'  {nm:<18}  {r["bpm"]:>7.2f}  {err:>7.2f}  '
              f'{r["time_ms"]:>10.1f}  {flg} {strat}')
    print('='*64 + '\n')
    print(f'完成 → {OUT_DIR}\n')


if __name__ == '__main__':
    main()
