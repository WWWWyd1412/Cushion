# -*- coding: utf-8 -*-
"""
呼吸算法全面改进版 v4.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
对全部10种算法统一应用两项改进:
  改进A: 多ROI ICA融合信号作为算法输入
  改进B: 自相关AutoCorr-BPM代替FPR峰值计数

输出:
  - 每种算法单独一张对比图 (2面板: 滑窗BPM曲线 + 信号波形)
  - BPM汇总对比图 (改进前 vs 改进后)
  - 改进结果汇总.csv
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
    extract_breath_mean, extract_breath_acmd,
    extract_breath_vmd,  extract_breath_emd,
    extract_breath_afd,  extract_breath_vmd_mape,
    extract_breath_goa_vmd, extract_breath_smvmd,
    extract_breath_mvmd, extract_breath_multi_roi_ica,
)

# ── 配置 ────────────────────────────────────────────────────────
FS        = 11.2
TRIM_SEC  = 20.0
ROI_SIZE  = 3
K_ROIS    = 4
MIN_DIST  = 5
WIN_SEC   = 30.0
STEP_SEC  = 5.0
DEADZONE  = 30
CLIP_MAX  = 2000

DATA_FILE = os.path.join(PROJECT_ROOT, 'data',         '20260702_160410_40x40.txt')
REF_FILE  = os.path.join(PROJECT_ROOT, 'Precise_Data', '刘若红0702.txt')
OUT_DIR   = os.path.join(PROJECT_ROOT, 'Contrast', '呼吸', '刘若红_0702_160410_v4全面改进')

ALGO_NAMES = [
    '均值法', 'ACMD', 'VMD', 'EMD', 'AFD',
    'VMD-MAPE', 'GOA-VMD', 'SMVMD', 'MVMD', 'Multi-ROI ICA',
]

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
# 工具: AutoCorr-BPM
# ════════════════════════════════════════════════════════════════
def bpm_autocorr(signal: np.ndarray, fs: float,
                 min_bpm=6.0, max_bpm=40.0) -> float:
    n = len(signal)
    if n < 20:
        return 0.0
    s   = signal - signal.mean()
    acf = correlate(s, s, mode='full')[n-1:]
    acf = acf / (acf[0] + 1e-12)

    lag_min = max(1, int(60.0 / max_bpm * fs))
    lag_max = min(n-1, int(60.0 / min_bpm * fs))
    if lag_min >= lag_max:
        return 0.0

    seg   = acf[lag_min:lag_max]
    peaks, props = find_peaks(seg, prominence=0.08)
    if len(peaks) == 0:
        peak = lag_min + int(np.argmax(seg))
    else:
        peak = lag_min + int(peaks[np.argmax(props['prominences'])])

    bpm = 60.0 / (peak / fs)
    return float(bpm) if min_bpm <= bpm <= max_bpm else 0.0


def both_bpm(sig: np.ndarray, fs: float) -> tuple:
    """返回 (FPR-BPM, AutoCorr-BPM)"""
    return (calculate_bpm_fpr(sig, fs=fs, min_dist_s=1.5),
            bpm_autocorr(sig, fs=fs))


# ════════════════════════════════════════════════════════════════
# 工具: 多ROI ICA融合
# ════════════════════════════════════════════════════════════════
def ica_fuse(roi_mat: np.ndarray, fs: float) -> np.ndarray:
    """
    roi_mat: (M, N)  → 返回呼吸频段SNR最高的独立分量 (N,)
    失败时自动回退到PCA主分量。
    """
    from sklearn.decomposition import FastICA, PCA
    from scipy.fft import fft, fftfreq

    M, N = roi_mat.shape
    if M == 1:
        return roi_mat[0]

    X = roi_mat.T                      # (N, M)
    n_c = min(M, 5)

    try:
        src = FastICA(n_components=n_c, random_state=42,
                      max_iter=2000, tol=1e-3).fit_transform(X)
    except Exception:
        src = PCA(n_components=n_c, random_state=42).fit_transform(X)

    freqs = fftfreq(N, 1.0/fs)[:N//2]
    in_b  = (freqs >= 0.1) & (freqs <= 0.5)
    best, best_snr = None, -999.0
    for k in range(src.shape[1]):
        c   = src[:, k]
        psd = np.abs(fft(c))[:N//2]**2
        snr = 10*np.log10(psd[in_b].sum() / (psd[~in_b & (freqs>0)].sum()+1e-9))
        if snr > best_snr:
            best_snr, best = snr, c

    mean_s = roi_mat.mean(axis=0)
    return best if np.dot(best, mean_s) >= 0 else -best


# ════════════════════════════════════════════════════════════════
# 数据加载
# ════════════════════════════════════════════════════════════════
def load_cushion(filepath):
    print(f"[DATA] {os.path.basename(filepath)}")
    frames = []
    with open(filepath, 'r', encoding='utf-8') as fh:
        for line in fh:
            p = line.split()
            if len(p) < 1601:
                continue
            try:
                t  = datetime.strptime(p[0], '%H:%M:%S.%f')
                ts = t.hour*3600 + t.minute*60 + t.second + t.microsecond/1e6
            except ValueError:
                continue
            raw = np.array(p[1:1601], dtype=np.float32).reshape(40,40)
            f   = np.clip(raw, 0, CLIP_MAX).astype(np.float32)
            f[f < DEADZONE] = 0
            f = median_filter(f, size=3)
            f = gaussian_filter(f, sigma=0.5)
            frames.append(f)

    frames = np.array(frames, dtype=np.float32)
    trim   = int(TRIM_SEC * FS)
    if len(frames) > 2*trim:
        frames = frames[trim:-trim]
    print(f"       {len(frames)} 帧  ({len(frames)/FS:.1f}s)")
    return frames


def load_ref(filepath):
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
    ds_idx = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith('CH1'):
            ds_idx = i+2; break
    raw = []
    for ln in lines[ds_idx:]:
        try:
            raw.append(float(ln.strip().split('\t')[0]))
        except (ValueError, IndexError):
            continue
    raw = np.array(raw, dtype=np.float64)
    trim = int(TRIM_SEC * fs_ref)
    if len(raw) > 2*trim:
        raw = raw[trim:-trim]
    raw -= raw.mean()
    b, a    = butter(4, 1.0/(0.5*fs_ref), btype='low')
    raw_lp  = filtfilt(b, a, raw)
    ds      = max(1, int(fs_ref/10.0))
    sig_ds  = raw_lp[::ds]
    fs_ds   = fs_ref / ds
    sig_bp  = butter_bandpass_filter(sig_ds, 0.1, 0.5, fs=fs_ds, order=4)
    fpr_b, acr_b = both_bpm(sig_bp, fs_ds)
    print(f"       FPR={fpr_b:.2f}  ACR={acr_b:.2f} BPM")
    return sig_bp, float(fs_ds), float(fpr_b), float(acr_b)


# ════════════════════════════════════════════════════════════════
# ROI选取 & 信号构建
# ════════════════════════════════════════════════════════════════
def _find_split(mean_frame):
    col_sum = mean_frame.sum(axis=0)
    return 12 + int(np.argmin(col_sum[12:28]))


def _pick_centers(zone_frame, k, min_d, c_offset):
    order = np.argsort(zone_frame.ravel())[::-1]
    centers = []
    for idx in order:
        r, c_l = np.unravel_index(idx, zone_frame.shape)
        c = c_l + c_offset
        if not any(max(abs(r-cr), abs(c-cc)) < min_d for cr,cc in centers):
            centers.append((r, c))
        if len(centers) == k:
            break
    while len(centers) < k:
        centers.append((zone_frame.shape[0]//2, c_offset + zone_frame.shape[1]//2))
    return centers


def build_roi_matrix(frames):
    """返回 (rois列表, mean_frame, roi_mat(M,N))"""
    mf    = frames.mean(axis=0)
    split = _find_split(mf)
    print(f"[ROI]  自适应分割列={split}")

    lc = _pick_centers(mf[:, :split],       K_ROIS, MIN_DIST, 0)
    rc = _pick_centers(mf[:, split:],        K_ROIS, MIN_DIST, split)
    rois = ([{'label':f'L{i+1}','center':c,'side':'left'}  for i,c in enumerate(lc)] +
            [{'label':f'R{i+1}','center':c,'side':'right'} for i,c in enumerate(rc)])

    half = ROI_SIZE // 2
    H, W = frames.shape[1], frames.shape[2]
    sigs = []
    for roi in rois:
        r, c = roi['center']
        rs, re = max(0,r-half), min(H,r+half+1)
        cs, ce = max(0,c-half), min(W,c+half+1)
        ts = frames[:, rs:re, cs:ce].mean(axis=(1,2))
        ts -= ts.mean()
        ts  = wavelet_denoise(ts, alpha=0.5)
        ts  = butter_bandpass_filter(ts, 0.1, 0.5, fs=FS, order=3)
        sigs.append(ts.astype(np.float64))
        print(f"       {roi['label']}: 行{r:2d} 列{c:2d}  "
              f"压力={mf[r,c]:.1f}")

    return rois, mf, np.array(sigs)   # roi_mat: (M, N)


# ════════════════════════════════════════════════════════════════
# 全部算法统一执行
# 每种算法返回:
#   orig_sig  : 原始方式提取的信号 (单ROI或全帧)
#   impr_sig  : 改进方式信号 (ICA融合 1D 输入)
#   orig_fpr/acr, impr_fpr/acr : 对应 BPM
# ════════════════════════════════════════════════════════════════
def _best_roi_sig(roi_mat):
    """能量最大的单ROI信号"""
    return roi_mat[int(np.argmax([np.std(s) for s in roi_mat]))]


def run_all(roi_mat: np.ndarray, frames: np.ndarray,
            fused: np.ndarray) -> dict:
    """
    roi_mat : (M, N) 各ROI信号
    frames  : (N,40,40) 预处理后帧序列
    fused   : (N,) ICA融合信号
    返回 {algo_name: {orig_sig, impr_sig,
                      orig_fpr, orig_acr,
                      impr_fpr, impr_acr}}
    """
    best1d = _best_roi_sig(roi_mat)   # 原始1D信号（能量最大ROI）

    def _run(fn_orig, fn_impr, label):
        print(f"    {label}...", end='', flush=True)
        try:
            os = fn_orig()
            of, oa = both_bpm(os, FS)
        except Exception as e:
            print(f" [原始失败:{e}]", end='')
            os = np.zeros_like(fused); of = oa = 0.0
        try:
            ims = fn_impr()
            imf, ima = both_bpm(ims, FS)
        except Exception as e:
            print(f" [改进失败:{e}]", end='')
            ims = np.zeros_like(fused); imf = ima = 0.0
        print(f"  原始FPR={of:.2f}  改进ACR={ima:.2f}")
        return dict(orig_sig=os, impr_sig=ims,
                    orig_fpr=of, orig_acr=oa,
                    impr_fpr=imf, impr_acr=ima)

    results = {}

    # ── 1D算法: 原始=单最优ROI, 改进=ICA融合 ──
    results['均值法'] = _run(
        lambda: extract_breath_mean(best1d),
        lambda: extract_breath_mean(fused),
        '均值法')
    results['ACMD'] = _run(
        lambda: extract_breath_acmd(best1d, fs=FS),
        lambda: extract_breath_acmd(fused,  fs=FS),
        'ACMD')
    results['VMD'] = _run(
        lambda: extract_breath_vmd(best1d, fs=FS),
        lambda: extract_breath_vmd(fused,  fs=FS),
        'VMD')
    results['EMD'] = _run(
        lambda: extract_breath_emd(best1d, fs=FS),
        lambda: extract_breath_emd(fused,  fs=FS),
        'EMD')
    results['AFD'] = _run(
        lambda: extract_breath_afd(best1d, fs=FS),
        lambda: extract_breath_afd(fused,  fs=FS),
        'AFD')

    # ── 3D算法: 原始=全帧, 改进=ICA融合1D ──
    results['VMD-MAPE'] = _run(
        lambda: extract_breath_vmd_mape(frames,  fs=FS),
        lambda: extract_breath_vmd_mape(fused,   fs=FS),
        'VMD-MAPE')
    results['GOA-VMD'] = _run(
        lambda: extract_breath_goa_vmd(frames,   fs=FS),
        lambda: extract_breath_goa_vmd(fused,    fs=FS),
        'GOA-VMD')
    results['SMVMD'] = _run(
        lambda: extract_breath_smvmd(frames,     fs=FS),
        lambda: extract_breath_smvmd(fused,      fs=FS),
        'SMVMD')
    results['MVMD'] = _run(
        lambda: extract_breath_mvmd(frames,      fs=FS),
        lambda: extract_breath_mvmd(fused,       fs=FS),
        'MVMD')
    results['Multi-ROI ICA'] = _run(
        lambda: extract_breath_multi_roi_ica(frames, fs=FS),
        lambda: extract_breath_multi_roi_ica(fused,  fs=FS),
        'Multi-ROI ICA')

    return results


# ════════════════════════════════════════════════════════════════
# 滑动窗口 BPM（返回 FPR + AutoCorr 两条曲线）
# ════════════════════════════════════════════════════════════════
def sw_bpm(sig, fs):
    win, step = int(WIN_SEC*fs), int(STEP_SEC*fs)
    n = len(sig)
    times, fpr_b, acr_b = [], [], []
    i = 0
    while i + win <= n:
        seg = sig[i:i+win]
        times.append((i + win/2) / fs)
        fpr_b.append(calculate_bpm_fpr(seg, fs=fs, min_dist_s=1.5))
        acr_b.append(bpm_autocorr(seg, fs=fs))
        i += step
    return np.array(times), np.array(fpr_b), np.array(acr_b)


# ════════════════════════════════════════════════════════════════
# 单算法独立图：2面板（滑窗BPM + 信号波形）
# ════════════════════════════════════════════════════════════════
def plot_single_algo(name: str, res: dict,
                     ref_sig: np.ndarray, ref_fs: float,
                     ref_fpr: float, ref_acr: float,
                     out_dir: str):
    """
    为单个算法生成一张完整的对比图:
    面板1 (上): 滑动窗口BPM时间曲线
      - 红色: 参考RSP  (实线=FPR, 虚线=AutoCorr)
      - 蓝色: 原始方法 (实线=FPR, 虚线=AutoCorr)
      - 绿色: ICA改进  (实线=FPR, 虚线=AutoCorr)
    面板2 (下): 提取信号波形（标准差归一化后叠加）
      - 红  : 参考RSP
      - 蓝  : 原始提取信号
      - 绿  : ICA改进信号
    """
    fig, (ax_bpm, ax_sig) = plt.subplots(
        2, 1, figsize=(13, 8), constrained_layout=True
    )
    title = (f'[{name}]  '
             f'参考: FPR={ref_fpr:.2f} / ACR={ref_acr:.2f} BPM\n'
             f'原始: FPR={res["orig_fpr"]:.2f} / ACR={res["orig_acr"]:.2f}    '
             f'改进: FPR={res["impr_fpr"]:.2f} / ACR={res["impr_acr"]:.2f}')
    fig.suptitle(title, fontsize=11)

    # ── 面板1: 滑窗BPM ──
    rt, rf, ra = sw_bpm(ref_sig,        ref_fs)
    ot, of, oa = sw_bpm(res['orig_sig'], FS)
    it, if_, ia = sw_bpm(res['impr_sig'], FS)

    ax_bpm.plot(rt, rf, 'r-',  lw=2.2, label=f'参考-FPR  ({ref_fpr:.1f})')
    ax_bpm.plot(rt, ra, 'r--', lw=1.8, label=f'参考-ACR  ({ref_acr:.1f})',
                alpha=0.7)
    ax_bpm.plot(ot, of, 'b-',  lw=1.6,
                label=f'原始-FPR  ({res["orig_fpr"]:.1f})')
    ax_bpm.plot(ot, oa, 'b--', lw=1.4,
                label=f'原始-ACR  ({res["orig_acr"]:.1f})', alpha=0.7)
    ax_bpm.plot(it, if_, color='#27ae60', lw=1.6,
                label=f'ICA改进-FPR  ({res["impr_fpr"]:.1f})')
    ax_bpm.plot(it, ia,  color='#27ae60', lw=1.4, ls='--',
                label=f'ICA改进-ACR  ({res["impr_acr"]:.1f})', alpha=0.8)

    ax_bpm.axhline(ref_fpr, color='red',      lw=0.8, ls=':', alpha=0.5)
    ax_bpm.set_ylabel('BPM');  ax_bpm.set_xlabel('时间 (s)')
    ax_bpm.set_title('滑动窗口BPM  (实线=FPR, 虚线=AutoCorr)')
    ax_bpm.legend(fontsize=8, ncol=2, loc='upper right')
    ax_bpm.grid(alpha=0.25)
    ax_bpm.set_ylim(0, max(40, ref_fpr * 2.8))

    # ── 面板2: 信号波形 ──
    def _norm(s):
        std = np.std(s)
        return s / std if std > 1e-9 else s

    t_ref  = np.arange(len(ref_sig))        / ref_fs
    t_orig = np.arange(len(res['orig_sig'])) / FS
    t_impr = np.arange(len(res['impr_sig'])) / FS

    ax_sig.plot(t_ref,  _norm(ref_sig),        'r-',
                lw=1.5, alpha=0.85, label='参考RSP (归一)')
    ax_sig.plot(t_orig, _norm(res['orig_sig']), 'b-',
                lw=1.2, alpha=0.75, label='原始信号 (归一)')
    ax_sig.plot(t_impr, _norm(res['impr_sig']), color='#27ae60',
                lw=1.2, alpha=0.85, label='ICA改进 (归一)')

    ax_sig.set_ylabel('归一化幅值');  ax_sig.set_xlabel('时间 (s)')
    ax_sig.set_title('提取信号波形（标准差归一化）')
    ax_sig.legend(fontsize=9, loc='upper right')
    ax_sig.grid(alpha=0.25)
    ax_sig.set_ylim(-5, 5)

    safe = name.replace(' ', '_').replace('-', '_')
    path = os.path.join(out_dir, f'{safe}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {name:<16} → {os.path.basename(path)}")


# ════════════════════════════════════════════════════════════════
# 汇总对比图：改进前 vs 改进后 (AutoCorr) 绝对误差
# ════════════════════════════════════════════════════════════════
def plot_summary(all_results: dict, ref_fpr: float, ref_acr: float,
                 out_dir: str):
    names  = list(all_results.keys())
    n      = len(names)
    x      = np.arange(n)

    orig_err = [abs(all_results[nm]['orig_fpr'] - ref_fpr) for nm in names]
    impr_err = [abs(all_results[nm]['impr_acr'] - ref_acr) for nm in names]

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    w = 0.35
    b1 = ax.bar(x - w/2, orig_err, w, label='原始 (FPR)',
                color='#5b9bd5', alpha=0.85)
    b2 = ax.bar(x + w/2, impr_err, w, label='ICA改进 (AutoCorr)',
                color='#70ad47', alpha=0.85)

    for bar, v in zip(b1, orig_err):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f'{v:.1f}', ha='center', va='bottom', fontsize=8, color='#2e6da4')
    for bar, v in zip(b2, impr_err):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f'{v:.1f}', ha='center', va='bottom', fontsize=8, color='#507e32')

    ax.axhline(1.5, color='green',  lw=1.2, ls='--', alpha=0.7, label='±1.5 BPM阈值')
    ax.axhline(3.0, color='orange', lw=1.2, ls='--', alpha=0.7, label='±3.0 BPM阈值')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha='right')
    ax.set_ylabel('|BPM误差|')
    ax.set_title('全算法改进前后绝对误差对比  (参考RSP CH1)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis='y')

    path = os.path.join(out_dir, '汇总误差对比.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  汇总图 → {path}")


# ════════════════════════════════════════════════════════════════
# CSV
# ════════════════════════════════════════════════════════════════
def save_csv(all_results: dict, ref_fpr: float, ref_acr: float,
             out_dir: str):
    path = os.path.join(out_dir, '改进结果汇总.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['算法',
                    '原始-FPR', '原始-ACR', '原始FPR误差', '原始ACR误差',
                    '改进-FPR', '改进-ACR', '改进FPR误差', '改进ACR误差',
                    '参考-FPR', '参考-ACR'])
        for nm, r in all_results.items():
            w.writerow([
                nm,
                f"{r['orig_fpr']:.3f}", f"{r['orig_acr']:.3f}",
                f"{abs(r['orig_fpr']-ref_fpr):.3f}",
                f"{abs(r['orig_acr']-ref_acr):.3f}",
                f"{r['impr_fpr']:.3f}", f"{r['impr_acr']:.3f}",
                f"{abs(r['impr_fpr']-ref_fpr):.3f}",
                f"{abs(r['impr_acr']-ref_acr):.3f}",
                f"{ref_fpr:.3f}", f"{ref_acr:.3f}",
            ])
    print(f"  CSV → {path}")


# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print('\n' + '='*62)
    print('  呼吸算法全面改进版 v4.0')
    print(f'  输出: {OUT_DIR}')
    print('='*62)

    # 1. 加载数据
    frames                          = load_cushion(DATA_FILE)
    ref_sig, ref_fs, ref_fpr, ref_acr = load_ref(REF_FILE)

    # 2. ROI选取 & ICA融合
    rois, mf, roi_mat = build_roi_matrix(frames)
    fused = ica_fuse(roi_mat, FS)
    print(f"       ICA融合信号: {fused.shape}")

    # 3. 运行全部算法
    print('\n[ALGO]')
    all_res = run_all(roi_mat, frames, fused)

    # 4. 每种算法单独绘图
    print('\n[PLOT] 单算法图表:')
    for name in ALGO_NAMES:
        plot_single_algo(name, all_res[name],
                         ref_sig, ref_fs, ref_fpr, ref_acr, OUT_DIR)

    # 5. 汇总图 + CSV
    print('\n[PLOT] 汇总对比图:')
    plot_summary(all_res, ref_fpr, ref_acr, OUT_DIR)
    print('\n[CSV]')
    save_csv(all_res, ref_fpr, ref_acr, OUT_DIR)

    # 6. 控制台汇总
    print('\n' + '='*62)
    print(f'  参考: FPR={ref_fpr:.2f}  ACR={ref_acr:.2f} BPM')
    print(f'  {"算法":<16}  {"原始FPR":>8}  {"原始err":>8}  '
          f'{"改进ACR":>8}  {"改进err":>8}')
    print(f'  {"-"*56}')
    for nm in ALGO_NAMES:
        r   = all_res[nm]
        e_o = abs(r['orig_fpr'] - ref_fpr)
        e_i = abs(r['impr_acr'] - ref_acr)
        flg = 'OK' if e_i<=1.5 else ('~' if e_i<=3 else 'X')
        imp = '↑' if e_i < e_o else ('=' if abs(e_i-e_o)<0.1 else '↓')
        print(f'  {nm:<16}  {r["orig_fpr"]:>8.2f}  {e_o:>8.2f}  '
              f'{r["impr_acr"]:>8.2f}  {e_i:>8.2f}  {flg} {imp}')
    print('='*62 + '\n')
    print(f'完成 → {OUT_DIR}\n')


if __name__ == '__main__':
    main()

