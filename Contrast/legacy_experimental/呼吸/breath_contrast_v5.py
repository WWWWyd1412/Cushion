# -*- coding: utf-8 -*-
"""
呼吸算法 v5.0 — 三估计器投票 + AFD修复 + 多项式去趋势
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
在 v4.0 的基础上新增三项改进:

  改进C: FFT-BPM (频谱主频估计)
         直接在0.1-0.5Hz呼吸频段找功率谱密度最大值
         → 三种BPM估计 (FPR / ACR / FFT) 取中位数作为最终结果

  改进D: AFD算法修复
         原AFD用20个固定频率扫描(分辨率~0.021Hz), 导致误差大
         改为: 先用FFT找到信号中主频, 再以该主频提取单分量

  改进E: 多项式去趋势 (poly-detrend, 阶数=3)
         ICA融合前先对每个ROI信号做多项式去趋势
         消除坐姿缓慢漂移对信号的低频污染

输出结构:
  Contrast/刘若红_0702_160410_v5改进/
    ├── {算法名}.png      (每算法单独一张: 三BPM曲线 + 信号波形)
    ├── 汇总误差对比.png
    └── 改进结果汇总.csv
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys, os, csv, warnings
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
from scipy.signal import butter, filtfilt, find_peaks, correlate, detrend

from algorithms.base import (
    calculate_bpm_fpr, butter_bandpass_filter, wavelet_denoise,
    select_best_component,
)
from algorithms.breath_extract import (
    extract_breath_mean, extract_breath_acmd,
    extract_breath_vmd,  extract_breath_emd,
    extract_breath_vmd_mape, extract_breath_goa_vmd,
    extract_breath_smvmd, extract_breath_mvmd,
    extract_breath_multi_roi_ica,
)

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
OUT_DIR   = os.path.join(ROOT, 'Contrast', '呼吸', '刘若红_0702_160410_v5改进')

ALGO_NAMES = [
    '均值法', 'ACMD', 'VMD', 'EMD', 'AFD(改)',
    'VMD-MAPE', 'GOA-VMD', 'SMVMD', 'MVMD', 'Multi-ROI ICA',
]

def _font():
    for n in ['SimHei','Microsoft YaHei','WenQuanYi Micro Hei']:
        try:
            fm.findfont(fm.FontProperties(family=n), fallback_to_default=False)
            plt.rcParams.update({'font.family': n, 'axes.unicode_minus': False})
            return
        except Exception:
            pass
    plt.rcParams['axes.unicode_minus'] = False

_font()


# ════════════════════════════════════════════════════════════════
# 改进C: 三种BPM估计器
# ════════════════════════════════════════════════════════════════
def bpm_fpr(sig, fs):
    """FPR: 峰值计数法"""
    return float(calculate_bpm_fpr(sig, fs=fs, min_dist_s=1.5))


def bpm_acr(sig, fs, min_bpm=6.0, max_bpm=40.0):
    """ACR: 自相关基础周期法"""
    n = len(sig)
    if n < 20:
        return 0.0
    s   = sig - sig.mean()
    acf = correlate(s, s, mode='full')[n-1:]
    acf = acf / (acf[0] + 1e-12)
    lg_min = max(1, int(60.0/max_bpm*fs))
    lg_max = min(n-1, int(60.0/min_bpm*fs))
    if lg_min >= lg_max:
        return 0.0
    seg = acf[lg_min:lg_max]
    pks, pr = find_peaks(seg, prominence=0.08)
    pk = (lg_min + int(pks[np.argmax(pr['prominences'])])
          if len(pks) else lg_min + int(np.argmax(seg)))
    bpm = 60.0 / (pk / fs)
    return float(bpm) if min_bpm <= bpm <= max_bpm else 0.0


def bpm_fft(sig, fs, min_bpm=6.0, max_bpm=40.0):
    """
    FFT-BPM: 频谱主频法
    在0.1–0.5 Hz (即 6–30 BPM) 的功率谱中找最大峰对应频率。
    对稳定周期信号最准；不受波形不对称影响。
    """
    n     = len(sig)
    freqs = np.fft.rfftfreq(n, 1.0/fs)
    psd   = np.abs(np.fft.rfft(sig - sig.mean()))**2
    f_lo, f_hi = min_bpm/60.0, max_bpm/60.0
    mask  = (freqs >= f_lo) & (freqs <= f_hi)
    if not mask.any():
        return 0.0
    dom   = freqs[mask][np.argmax(psd[mask])]
    return float(dom * 60.0)


def vote_bpm(sig, fs):
    """
    三方投票: 取 FPR / ACR / FFT 三个估计的中位数。
    零值（估计失败）不参与中位数计算。
    """
    vals = [bpm_fpr(sig, fs), bpm_acr(sig, fs), bpm_fft(sig, fs)]
    valid = [v for v in vals if v > 0]
    if not valid:
        return 0.0, vals
    return float(np.median(valid)), vals


def triple_bpm(sig, fs):
    """返回 (fpr, acr, fft, vote) 四个BPM值"""
    f = bpm_fpr(sig, fs)
    a = bpm_acr(sig, fs)
    x = bpm_fft(sig, fs)
    valid = [v for v in [f, a, x] if v > 0]
    v = float(np.median(valid)) if valid else 0.0
    return f, a, x, v


# ════════════════════════════════════════════════════════════════
# 改进D: AFD修复版 — FFT主频初始化
# ════════════════════════════════════════════════════════════════
def extract_breath_afd_v2(signal: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    修复版 AFD (Adaptive Fourier Decomposition):
      1. 用 FFT 在呼吸频段 (0.08–0.6 Hz) 找主频 f0
      2. 以 f0 为中心，在 ±0.05 Hz 范围内细化搜索 (100点)
      3. 用最终频率的余弦/正弦分量做单模态投影提取

    原版问题: linspace(0.1,0.5,20) 分辨率~0.021Hz,
    且靠 select_best_component 选择, 极易选到次优频率。
    """
    n = len(signal)
    if n < 30:
        return signal.copy()

    sig = signal - signal.mean()

    # Step1: FFT找粗主频
    freqs_all = np.fft.rfftfreq(n, 1.0/fs)
    psd_all   = np.abs(np.fft.rfft(sig))**2
    mask      = (freqs_all >= 0.08) & (freqs_all <= 0.6)
    if not mask.any():
        return sig
    f_coarse = freqs_all[mask][np.argmax(psd_all[mask])]

    # Step2: 精细搜索 f0±0.05 Hz
    f_lo  = max(0.08, f_coarse - 0.05)
    f_hi  = min(0.60, f_coarse + 0.05)
    f_cands = np.linspace(f_lo, f_hi, 100)
    t = np.arange(n) / fs

    best_f, best_e = f_coarse, -1.0
    for f in f_cands:
        c = np.cos(2*np.pi*f*t)
        s = np.sin(2*np.pi*f*t)
        # 投影能量
        e = (np.dot(sig, c)**2 + np.dot(sig, s)**2) / n
        if e > best_e:
            best_e, best_f = e, f

    # Step3: 单模态提取
    c = np.cos(2*np.pi*best_f*t)
    s = np.sin(2*np.pi*best_f*t)
    comp = (c * np.dot(sig, c) / (np.dot(c, c)+1e-9) +
            s * np.dot(sig, s) / (np.dot(s, s)+1e-9))
    return comp


# ════════════════════════════════════════════════════════════════
# 改进E: 多项式去趋势
# ════════════════════════════════════════════════════════════════
def poly_detrend(sig: np.ndarray, order: int = 3) -> np.ndarray:
    """
    多项式去趋势: 拟合 order 阶多项式并减去。
    去除因坐姿调整、传感器漂移引起的低频缓慢趋势。
    比简单去均值更彻底，比高通滤波不引入边界振铃。
    """
    t = np.arange(len(sig), dtype=np.float64)
    coeffs = np.polyfit(t, sig, order)
    trend  = np.polyval(coeffs, t)
    return sig - trend


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
                t = datetime.strptime(p[0],'%H:%M:%S.%f')
                # timestamp not stored, only frames needed
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
    print(f"       {len(frames)} 帧 ({len(frames)/FS:.1f}s)")
    return frames


def load_ref(fp):
    print(f"[REF]  {os.path.basename(fp)}")
    fs_r = 2000.0
    with open(fp,'r',encoding='utf-8',errors='replace') as fh:
        lines = fh.readlines()
    for ln in lines[:4]:
        if 'msec/sample' in ln:
            try: fs_r = 1000.0/float(ln.strip().split()[0])
            except: pass
            break
    di = 0
    for i,ln in enumerate(lines):
        if ln.strip().startswith('CH1'): di = i+2; break
    raw = []
    for ln in lines[di:]:
        try: raw.append(float(ln.strip().split('\t')[0]))
        except: continue
    raw = np.array(raw, dtype=np.float64)
    trim = int(TRIM_SEC*fs_r)
    if len(raw) > 2*trim: raw = raw[trim:-trim]
    raw -= raw.mean()
    b,a  = butter(4, 1.0/(0.5*fs_r), btype='low')
    lp   = filtfilt(b, a, raw)
    ds   = max(1, int(fs_r/10.0))
    sig  = lp[::ds]
    fs_d = fs_r/ds
    sig  = butter_bandpass_filter(sig, 0.1, 0.5, fs=fs_d, order=4)
    f,a_,x,v = triple_bpm(sig, fs_d)
    print(f"       FPR={f:.2f}  ACR={a_:.2f}  FFT={x:.2f}  Vote={v:.2f}")
    return sig, float(fs_d), f, a_, x, v


# ════════════════════════════════════════════════════════════════
# ROI选取 + 信号矩阵（含多项式去趋势）
# ════════════════════════════════════════════════════════════════
def _split_col(mf):
    cs = mf.sum(axis=0)
    return 12 + int(np.argmin(cs[12:28]))


def _centers(zone, k, md, c_off):
    order = np.argsort(zone.ravel())[::-1]
    cens  = []
    for idx in order:
        r,cl = np.unravel_index(idx, zone.shape)
        c    = cl + c_off
        if not any(max(abs(r-cr),abs(c-cc))<md for cr,cc in cens):
            cens.append((r,c))
        if len(cens)==k: break
    while len(cens)<k: cens.append((zone.shape[0]//2, c_off+zone.shape[1]//2))
    return cens


def build_roi_matrix(frames):
    mf    = frames.mean(axis=0)
    sp    = _split_col(mf)
    print(f"[ROI]  分割列={sp}")
    lc = _centers(mf[:,:sp],   K_ROIS, MIN_DIST, 0)
    rc = _centers(mf[:,sp:],   K_ROIS, MIN_DIST, sp)
    rois = ([{'label':f'L{i+1}','c':c} for i,c in enumerate(lc)] +
            [{'label':f'R{i+1}','c':c} for i,c in enumerate(rc)])

    half = ROI_SIZE//2
    H,W  = frames.shape[1], frames.shape[2]
    sigs = []
    for roi in rois:
        r,c = roi['c']
        rs,re = max(0,r-half), min(H,r+half+1)
        cs,ce = max(0,c-half), min(W,c+half+1)
        ts = frames[:,rs:re,cs:ce].mean(axis=(1,2))
        # 改进E: 多项式去趋势 (order=3)
        ts = poly_detrend(ts, order=3)
        ts = wavelet_denoise(ts, alpha=0.5)
        ts = butter_bandpass_filter(ts, 0.1, 0.5, fs=FS, order=3)
        sigs.append(ts.astype(np.float64))
        print(f"       {roi['label']}: 行{r:2d}列{c:2d}  压力={mf[r,c]:.1f}")

    return rois, mf, np.array(sigs)


def ica_fuse(roi_mat, fs):
    from sklearn.decomposition import FastICA, PCA
    from scipy.fft import fft as _fft, fftfreq as _fftfreq
    M,N = roi_mat.shape
    if M==1: return roi_mat[0]
    X = roi_mat.T
    nc = min(M,5)
    try:
        src = FastICA(n_components=nc,random_state=42,
                      max_iter=2000,tol=1e-3).fit_transform(X)
    except:
        src = PCA(n_components=nc,random_state=42).fit_transform(X)
    freqs = _fftfreq(N,1.0/fs)[:N//2]
    inb   = (freqs>=0.1)&(freqs<=0.5)
    best,bsnr = None,-999.0
    for k in range(src.shape[1]):
        c   = src[:,k]
        psd = np.abs(_fft(c))[:N//2]**2
        snr = 10*np.log10(psd[inb].sum()/(psd[~inb&(freqs>0)].sum()+1e-9))
        if snr>bsnr: bsnr,best = snr,c
    ms = roi_mat.mean(axis=0)
    return best if np.dot(best,ms)>=0 else -best


# ════════════════════════════════════════════════════════════════
# 全部算法执行（ICA融合输入 + 三方投票BPM）
# ════════════════════════════════════════════════════════════════
def _run_one(name, fn, fused, frames):
    """执行单个算法, 返回 (sig_orig, sig_impr, fpr_i, acr_i, fft_i, vote_i)"""
    print(f"    {name}...", end='', flush=True)

    # 原始: 最优单ROI (能量最大) 走原版算法 -- 此处用frames全帧或fused 1D
    # 改进: ICA融合信号 + 三方投票
    try:
        sig = fn(fused, frames)
        f,a,x,v = triple_bpm(sig, FS)
        print(f"  FPR={f:.2f} ACR={a:.2f} FFT={x:.2f} Vote={v:.2f}")
        return sig, f, a, x, v
    except Exception as e:
        print(f"  [err:{e}]")
        return np.zeros(len(fused)), 0., 0., 0., 0.


def run_all(fused, frames):
    """
    统一接口: 所有算法均使用 ICA融合信号 (1D) 作为输入。
    3D算法的改进版走其1D输入分支。
    返回 {name: {'sig','fpr','acr','fft','vote'}}
    """
    algo_fns = {
        '均值法':      lambda s,f: extract_breath_mean(s),
        'ACMD':         lambda s,f: extract_breath_acmd(s, fs=FS),
        'VMD':          lambda s,f: extract_breath_vmd (s, fs=FS),
        'EMD':          lambda s,f: extract_breath_emd (s, fs=FS),
        'AFD(改)':      lambda s,f: extract_breath_afd_v2(s, fs=FS),
        'VMD-MAPE':     lambda s,f: extract_breath_vmd_mape(s,  fs=FS),
        'GOA-VMD':      lambda s,f: extract_breath_goa_vmd(s,   fs=FS),
        'SMVMD':        lambda s,f: extract_breath_smvmd(s,     fs=FS),
        'MVMD':         lambda s,f: extract_breath_mvmd(s,      fs=FS),
        'Multi-ROI ICA':lambda s,f: extract_breath_multi_roi_ica(s, fs=FS),
    }
    results = {}
    print('\n[ALGO]')
    for name in ALGO_NAMES:
        sig, f, a, x, v = _run_one(name, algo_fns[name], fused, frames)
        results[name] = {'sig':sig, 'fpr':f, 'acr':a, 'fft':x, 'vote':v}
    return results


# ════════════════════════════════════════════════════════════════
# 滑动窗口 — 三BPM曲线
# ════════════════════════════════════════════════════════════════
def sw_triple(sig, fs):
    win, step = int(WIN_SEC*fs), int(STEP_SEC*fs)
    n = len(sig)
    T, F, A, X, V = [], [], [], [], []
    i = 0
    while i+win <= n:
        seg = sig[i:i+win]
        f,a,x,v = triple_bpm(seg, fs)
        T.append((i+win/2)/fs)
        F.append(f); A.append(a); X.append(x); V.append(v)
        i += step
    return [np.array(arr) for arr in (T,F,A,X,V)]


# ════════════════════════════════════════════════════════════════
# 单算法独立图（3 BPM曲线 + 信号波形）
# ════════════════════════════════════════════════════════════════
def plot_one(name, res, ref_sig, ref_fs,
             ref_f, ref_a, ref_x, ref_v, out_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8),
                                    constrained_layout=True)
    rv = ref_v if ref_v > 0 else ref_f
    title = (
        f'[{name}]\n'
        f'参考: FPR={ref_f:.2f}  ACR={ref_a:.2f}  FFT={ref_x:.2f}  投票={rv:.2f} BPM\n'
        f'算法: FPR={res["fpr"]:.2f}  ACR={res["acr"]:.2f}  '
        f'FFT={res["fft"]:.2f}  投票={res["vote"]:.2f} BPM'
    )
    fig.suptitle(title, fontsize=10)

    # ── 面板1: 滑窗BPM ──
    rT,rF,rA,rX,rV = sw_triple(ref_sig, ref_fs)
    aT,aF,aA,aX,aV = sw_triple(res['sig'], FS)

    ax1.plot(rT, rF, 'r-',  lw=2.0, label=f'参考-FPR ({ref_f:.1f})')
    ax1.plot(rT, rA, 'r--', lw=1.6, label=f'参考-ACR ({ref_a:.1f})', alpha=0.75)
    ax1.plot(rT, rX, 'r:',  lw=1.5, label=f'参考-FFT ({ref_x:.1f})', alpha=0.6)
    ax1.plot(aT, aF, 'b-',  lw=1.6, label=f'算法-FPR ({res["fpr"]:.1f})')
    ax1.plot(aT, aA, 'b--', lw=1.4, label=f'算法-ACR ({res["acr"]:.1f})', alpha=0.8)
    ax1.plot(aT, aX, 'b:',  lw=1.3, label=f'算法-FFT ({res["fft"]:.1f})', alpha=0.7)
    ax1.plot(aT, aV, color='#27ae60', lw=2.0, label=f'算法-投票 ({res["vote"]:.1f})')

    ax1.axhline(rv, color='red', lw=0.8, ls=':', alpha=0.5)
    ax1.set_ylabel('BPM'); ax1.set_xlabel('时间 (s)')
    ax1.set_title('滑动窗口BPM  (实=FPR  虚=ACR  点=FFT  粗绿=投票)')
    ax1.legend(fontsize=7, ncol=4, loc='upper right')
    ax1.grid(alpha=0.22); ax1.set_ylim(0, max(40, rv*2.8))

    # ── 面板2: 信号波形 ──
    def N(s): std=np.std(s); return s/std if std>1e-9 else s
    t_r = np.arange(len(ref_sig)) / ref_fs
    t_a = np.arange(len(res['sig'])) / FS

    ax2.plot(t_r, N(ref_sig),    'r-',            lw=1.5, alpha=0.85, label='参考RSP')
    ax2.plot(t_a, N(res['sig']), color='#2980b9', lw=1.3, alpha=0.80, label='ICA+算法')
    ax2.set_ylabel('归一化幅值'); ax2.set_xlabel('时间 (s)')
    ax2.set_title('提取信号波形（标准差归一化）')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.22); ax2.set_ylim(-5, 5)

    safe = name.replace(' ','_').replace('(','').replace(')','').replace('-','_')
    path = os.path.join(out_dir, f'{safe}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {name:<18} → {os.path.basename(path)}")


# ════════════════════════════════════════════════════════════════
# 汇总误差对比图
# ════════════════════════════════════════════════════════════════
def plot_summary(all_res, ref_v, out_dir):
    names = list(all_res.keys())
    n     = len(names)
    x     = np.arange(n)
    v4_err  = [abs(all_res[nm]['fpr'] - ref_v) for nm in names]   # 原始FPR误差
    v5_err  = [abs(all_res[nm]['vote'] - ref_v) for nm in names]  # v5投票误差

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    w  = 0.35
    b1 = ax.bar(x-w/2, v4_err, w, label='ICA+FPR (v4基准)', color='#5b9bd5', alpha=0.85)
    b2 = ax.bar(x+w/2, v5_err, w, label='ICA+三方投票 (v5)', color='#70ad47', alpha=0.85)
    for bar,v in zip(b1,v4_err):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f'{v:.1f}', ha='center', va='bottom', fontsize=8, color='#2e6da4')
    for bar,v in zip(b2,v5_err):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f'{v:.1f}', ha='center', va='bottom', fontsize=8, color='#375c1e')

    ax.axhline(1.5, color='green',  lw=1.2, ls='--', alpha=0.7, label='±1.5 BPM')
    ax.axhline(3.0, color='orange', lw=1.2, ls='--', alpha=0.7, label='±3.0 BPM')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha='right')
    ax.set_ylabel('|BPM误差|')
    ax.set_title(f'v4 vs v5 绝对误差对比  (参考投票BPM={ref_v:.2f})')
    ax.legend(fontsize=9); ax.grid(alpha=0.25, axis='y')

    path = os.path.join(out_dir, '汇总误差对比.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  汇总图 → {path}")


# ════════════════════════════════════════════════════════════════
# CSV
# ════════════════════════════════════════════════════════════════
def save_csv(all_res, ref_f, ref_a, ref_x, ref_v, out_dir):
    path = os.path.join(out_dir, 'v5改进结果汇总.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['算法','FPR','ACR','FFT','投票BPM',
                    'FPR误差','ACR误差','FFT误差','投票误差',
                    '参考-FPR','参考-ACR','参考-FFT','参考-投票'])
        for nm, r in all_res.items():
            w.writerow([nm,
                f"{r['fpr']:.3f}", f"{r['acr']:.3f}",
                f"{r['fft']:.3f}", f"{r['vote']:.3f}",
                f"{abs(r['fpr']-ref_f):.3f}", f"{abs(r['acr']-ref_a):.3f}",
                f"{abs(r['fft']-ref_x):.3f}", f"{abs(r['vote']-ref_v):.3f}",
                f"{ref_f:.3f}", f"{ref_a:.3f}", f"{ref_x:.3f}", f"{ref_v:.3f}",
            ])
    print(f"  CSV → {path}")


# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print('\n'+'='*62)
    print('  呼吸算法 v5.0  三估计器投票 + AFD修复 + 多项式去趋势')
    print(f'  输出: {OUT_DIR}')
    print('='*62)

    frames                        = load_cushion(DATA_FILE)
    ref_sig, ref_fs, rf,ra,rx,rv  = load_ref(REF_FILE)
    rois, mf, roi_mat             = build_roi_matrix(frames)
    fused                         = ica_fuse(roi_mat, FS)
    print(f"       ICA融合信号 shape={fused.shape}")

    all_res = run_all(fused, frames)

    print('\n[PLOT] 单算法图:')
    for name in ALGO_NAMES:
        plot_one(name, all_res[name], ref_sig, ref_fs, rf,ra,rx,rv, OUT_DIR)

    print('\n[PLOT] 汇总图:')
    plot_summary(all_res, rv if rv>0 else rf, OUT_DIR)

    print('\n[CSV]')
    save_csv(all_res, rf, ra, rx, rv, OUT_DIR)

    # 控制台汇总
    ref_bpm = rv if rv > 0 else rf
    print('\n'+'='*62)
    print(f'  参考: FPR={rf:.2f} ACR={ra:.2f} FFT={rx:.2f} 投票={rv:.2f}')
    print(f'  {"算法":<18}  {"FPR":>7} {"ACR":>7} {"FFT":>7} {"投票":>7}  {"投票误差":>8}')
    print(f'  {"-"*56}')
    for nm in ALGO_NAMES:
        r = all_res[nm]
        e = abs(r['vote'] - ref_bpm)
        fl = 'OK' if e<=1.5 else ('~' if e<=3 else 'X')
        print(f'  {nm:<18}  {r["fpr"]:>7.2f} {r["acr"]:>7.2f} '
              f'{r["fft"]:>7.2f} {r["vote"]:>7.2f}  {e:>8.2f}  {fl}')
    print('='*62+'\n')
    print(f'完成 → {OUT_DIR}\n')


if __name__ == '__main__':
    main()

