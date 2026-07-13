# -*- coding: utf-8 -*-
"""
全预处理后心跳信号时域+频域对比图（8受试者）
预处理流程: 帧处理 -> ROI -> ICA融合 -> VME基线去除 -> LS谐波去除 -> IIR陷波 -> 带通
"""
import sys, warnings, os, time as _t
warnings.filterwarnings('ignore')

_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_DIR)
sys.path.insert(0, os.path.join(ROOT, '40_40_Extraction_1'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.ndimage import median_filter, gaussian_filter
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.decomposition import FastICA, PCA
from datetime import datetime
from algorithms.base import butter_bandpass_filter, wavelet_denoise, VME_Core

for _n in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']:
    try:
        fm.findfont(fm.FontProperties(family=_n), fallback_to_default=False)
        plt.rcParams.update({'font.family': _n, 'axes.unicode_minus': False})
        break
    except Exception:
        pass

SUBJECTS    = ['lbx1','lbx2','wyd1','wyd2','xxr1','xxr2','zxc1','zxc2']
CUSHION_DIR = os.path.join(ROOT, '40_40_Cushion_Data')
PPG_DIR     = os.path.join(ROOT, 'PPGdataset')
OUT_DIR     = os.path.join(ROOT, 'Contrast', '心跳')
os.makedirs(OUT_DIR, exist_ok=True)

FS = 11.18
TRIM = 20
HB_LO, HB_HI = 0.8, 2.2


# ── 工具 ──────────────────────────────────────────────────────────
def poly_detrend(s, order=3):
    t = np.arange(len(s), dtype=np.float64)
    return s - np.polyval(np.polyfit(t, s, order), t)


def load_cushion(fp):
    frames = []
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            p = line.split()
            if len(p) < 1601:
                continue
            try:
                datetime.strptime(p[0], '%H:%M:%S.%f')
            except ValueError:
                continue
            raw = np.array(p[1:1601], dtype=np.float32).reshape(40, 40)
            x = raw.astype(np.float32)
            if x.mean() > 1000:
                x = 4095 - x
            x = np.clip(x, 0, 1200)
            x[x < 30] = 0
            x = median_filter(x, size=3)
            x = gaussian_filter(x, sigma=0.5)
            frames.append(x)
    frames = np.array(frames, dtype=np.float32)
    trim = int(TRIM * FS)
    return frames[trim:-trim] if len(frames) > 2 * trim else frames


def ica_fuse_hb(roi_mat):
    from scipy.fft import fft as _fft, fftfreq as _ff
    M, N = roi_mat.shape
    if M == 1:
        return roi_mat[0]
    X = roi_mat.T
    nc = min(M, 5)
    try:
        src = FastICA(n_components=nc, random_state=42,
                      max_iter=2000, tol=1e-3).fit_transform(X)
    except Exception:
        src = PCA(n_components=nc, random_state=42).fit_transform(X)
    freqs = _ff(N, 1.0 / FS)[:N // 2]
    inb = (freqs >= HB_LO) & (freqs <= HB_HI)
    best, bsnr = None, -999.0
    for k in range(src.shape[1]):
        c = src[:, k]
        psd = np.abs(_fft(c))[:N // 2] ** 2
        snr = 10 * np.log10(psd[inb].sum() / (psd[~inb & (freqs > 0)].sum() + 1e-9))
        if snr > bsnr:
            bsnr, best = snr, c
    ms = roi_mat.mean(axis=0)
    return best if np.dot(best, ms) >= 0 else -best


def build_hb_signal(frames):
    """完整心跳预处理管道，返回 (fused_signal, breath_freq)"""
    mf = frames.mean(axis=0)
    sp = 12 + int(np.argmin(mf.sum(axis=0)[12:28]))

    def pick_centers(zone, k, md, off):
        order = np.argsort(zone.ravel())[::-1]
        cens = []
        for idx in order:
            r, cl = np.unravel_index(idx, zone.shape)
            c = cl + off
            if not any(max(abs(r - cr), abs(c - cc)) < md for cr, cc in cens):
                cens.append((r, c))
            if len(cens) == k:
                break
        while len(cens) < k:
            cens.append((zone.shape[0] // 2, off + zone.shape[1] // 2))
        return cens

    lc = pick_centers(mf[:, :sp], 4, 5, 0)
    rc = pick_centers(mf[:, sp:], 4, 5, sp)
    H, W = frames.shape[1], frames.shape[2]
    sigs = []
    for r, c in lc + rc:
        ts = frames[:, max(0, r-1):min(H, r+2),
                       max(0, c-1):min(W, c+2)].mean(axis=(1, 2))
        ts = poly_detrend(ts)
        ts = wavelet_denoise(ts, alpha=0.3)
        sigs.append(ts.astype(np.float64))

    fused = ica_fuse_hb(np.array(sigs))

    # 估算呼吸基频
    n = len(fused)
    fr = np.fft.rfftfreq(n, 1.0 / FS)
    ps = np.abs(np.fft.rfft(fused - fused.mean())) ** 2
    vm = (fr >= 0.1) & (fr <= 0.5)
    bf = float(fr[vm][np.argmax(ps[vm])]) if vm.any() else 0.25

    # VME 去呼吸基线
    try:
        hb_m = (fr >= HB_LO) & (fr <= HB_HI)
        fh = float(fr[hb_m][np.argmax(ps[hb_m])]) if hb_m.any() else 1.2
        alpha = 1000.0 * np.exp(1.09 * ((fh - 1.25) / -0.5) ** 2) if fh <= 1.25 else 1000.0
        u = VME_Core(fused - fused.mean(), fs=FS, f_init=bf, alpha=alpha)
        fused = fused - u
    except Exception:
        pass

    # LS 去除 < HB_LO 谐波
    t2 = np.arange(len(fused)) / FS
    cols = []
    for k in range(1, 20):
        fh2 = bf * k
        if fh2 >= HB_LO - 0.02 or fh2 >= 0.5 * FS - 0.05:
            break
        cols += [np.cos(2 * np.pi * fh2 * t2), np.sin(2 * np.pi * fh2 * t2)]
    if cols:
        A = np.column_stack(cols)
        x, _, _, _ = np.linalg.lstsq(A, fused - fused.mean(), rcond=None)
        fused = fused - A @ x

    # 高Q IIR 陷波 (HB_LO~HB_HI 内谐波)
    nyq = 0.5 * FS
    for k in range(1, 12):
        fh3 = bf * k
        if fh3 < HB_LO:
            continue
        if fh3 >= HB_HI or fh3 >= nyq - 0.05:
            break
        w0 = fh3 / nyq
        if 0 < w0 < 1:
            try:
                b, a = iirnotch(w0, Q=40)
                fused = filtfilt(b, a, fused)
            except Exception:
                pass

    fused = butter_bandpass_filter(fused, HB_LO, HB_HI, fs=FS, order=4)
    return fused, bf


def load_ppg(fp):
    """加载PPG(CH2)参考信号，同时返回呼吸基频"""
    fs_r = 2000.0
    with open(fp, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    for ln in lines[:4]:
        if 'msec/sample' in ln:
            try:
                fs_r = 1000.0 / float(ln.strip().split()[0])
            except Exception:
                pass
            break
    di = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith('CH1'):
            di = i + 2
            break
    ch1r, ch2r = [], []
    for ln in lines[di:]:
        cols = ln.strip().split('\t')
        try:
            ch1r.append(float(cols[0]))
            ch2r.append(float(cols[1]))
        except Exception:
            continue
    trim = int(TRIM * fs_r)

    # CH1 → 呼吸基频
    ch1 = np.array(ch1r, dtype=np.float64)
    if len(ch1) > 2 * trim:
        ch1 = ch1[trim:-trim]
    ch1 -= ch1.mean()
    b, a = butter(4, 1.0 / (0.5 * fs_r), btype='low')
    ds1 = max(1, int(fs_r / 10))
    sig_r = filtfilt(b, a, ch1)[::ds1]
    fs_r2 = fs_r / ds1
    sig_r = butter_bandpass_filter(sig_r, 0.1, 0.5, fs=fs_r2, order=4)
    fr_r = np.fft.rfftfreq(len(sig_r), 1.0 / fs_r2)
    ps_r = np.abs(np.fft.rfft(sig_r)) ** 2
    rm = (fr_r >= 0.1) & (fr_r <= 0.5)
    bf = float(fr_r[rm][np.argmax(ps_r[rm])]) if rm.any() else 0.25

    # CH2 → PPG 带通 0.8~2.2 Hz
    ch2 = np.array(ch2r, dtype=np.float64)
    if len(ch2) > 2 * trim:
        ch2 = ch2[trim:-trim]
    ch2 -= ch2.mean()
    b2, a2 = butter(4, 5.0 / (0.5 * fs_r), btype='low')
    ds2 = max(1, int(fs_r / 50))
    ppg = filtfilt(b2, a2, ch2)[::ds2]
    fs_ppg = fs_r / ds2
    ppg = butter_bandpass_filter(ppg, HB_LO, HB_HI, fs=fs_ppg, order=4)

    # PPG 参考心率
    fr_p = np.fft.rfftfreq(len(ppg), 1.0 / fs_ppg) * 60
    ps_p = np.abs(np.fft.rfft(ppg)) ** 2
    mp = (fr_p >= 30) & (fr_p <= 150)
    ref_hb = float(fr_p[mp][np.argmax(ps_p[mp])]) if mp.any() else 0.0
    return ppg, fs_ppg, ref_hb, bf


def bpm_acr_ref(sig, fs, min_bpm=40.0, max_bpm=150.0):
    """ACR法估计参考心率（与batch_analysis保持一致）"""
    from scipy.signal import correlate, find_peaks
    n = len(sig)
    if n < 30:
        return 0.0
    s   = sig - sig.mean()
    acf = correlate(s, s, mode='full')[n-1:]
    acf = acf / (acf[0] + 1e-12)
    lmin = max(1, int(60.0/max_bpm * fs))
    lmax = min(n-1, int(60.0/min_bpm * fs))
    if lmin >= lmax:
        return 0.0
    seg = acf[lmin:lmax]
    pks, pr = find_peaks(seg, prominence=0.05)
    pk = (lmin + int(pks[np.argmax(pr['prominences'])])
          if len(pks) else lmin + int(np.argmax(seg)))
    b = 60.0 / (pk / fs)
    return float(b) if min_bpm <= b <= max_bpm else 0.0
def norm(s):
    std = np.std(s)
    return s / std if std > 1e-9 else s


# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('  全预处理后心跳信号时域+频域对比（8受试者）')
print('='*60)

results = {}
for subj in SUBJECTS:
    t0 = _t.perf_counter()
    frames = load_cushion(os.path.join(CUSHION_DIR, f'{subj}.txt'))
    fused, bf = build_hb_signal(frames)
    ppg, fs_ppg, ref_hb, _ = load_ppg(os.path.join(PPG_DIR, f'{subj}.txt'))

    n_c = len(fused)
    fr_c = np.fft.rfftfreq(n_c, 1.0 / FS) * 60
    ps_c = np.abs(np.fft.rfft(fused - fused.mean())) ** 2
    mc = (fr_c >= 30) & (fr_c <= 150)
    peak_c = float(fr_c[mc][np.argmax(ps_c[mc])]) if mc.any() else 0.0

    results[subj] = {
        'fused': fused, 'ppg': ppg, 'fs_ppg': fs_ppg,
        'ref_hb': ref_hb, 'peak_c': peak_c, 'bf': bf,
    }
    elapsed = (_t.perf_counter() - t0) * 1000
    err = abs(peak_c - ref_hb)
    flag = 'OK' if err <= 5 else ('~' if err <= 10 else 'X')
    print(f'  {subj}: 座垫={peak_c:.1f}  PPG={ref_hb:.1f}  '
          f'误差={err:.1f} BPM  {flag}  ({elapsed:.0f}ms)')

# ── 每受试者单独 2×2 图 ──────────────────────────────────────────
print('\n绘制单受试者对比图...')
for subj in SUBJECTS:
    r = results[subj]
    fused  = r['fused'];  ppg    = r['ppg']
    fs_ppg = r['fs_ppg']; ref_hb = r['ref_hb']; peak_c = r['peak_c']
    err    = abs(peak_c - ref_hb)
    flag   = 'OK' if err <= 5 else ('~' if err <= 10 else 'X')

    t_c = np.arange(len(fused)) / FS
    t_p = np.arange(len(ppg))   / fs_ppg

    n_c  = len(fused); n_p = len(ppg)
    fr_c = np.fft.rfftfreq(n_c, 1.0/FS)*60
    ps_c = np.abs(np.fft.rfft(fused - fused.mean()))**2
    fr_p = np.fft.rfftfreq(n_p, 1.0/fs_ppg)*60
    ps_p = np.abs(np.fft.rfft(ppg   - ppg.mean()))**2
    mc = (fr_c >= 30) & (fr_c <= 150)
    mp = (fr_p >= 30) & (fr_p <= 150)
    ps_cn = ps_c[mc] / ps_c[mc].max()
    ps_pn = ps_p[mp] / ps_p[mp].max()

    fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(
        f'[{subj}]  全预处理后心跳信号对比\n'
        f'预处理: ROI→ICA→VME→LS→IIR陌波→带通{HB_LO}–{HB_HI}Hz\n'
        f'座垫主峰={peak_c:.1f} BPM  |  PPG参考={ref_hb:.1f} BPM  |  '
        f'误差={err:.1f} BPM  {flag}',
        fontsize=10, fontweight='bold'
    )

    # [0,0] 座垫时域
    axes[0, 0].plot(t_c, norm(fused), color='#2980b9', lw=1.0, alpha=0.9)
    axes[0, 0].set_title('座垫压力  时域波形（归一化）')
    axes[0, 0].set_xlabel('时间 (s)')
    axes[0, 0].set_ylabel('归一化幅値')
    axes[0, 0].grid(alpha=0.25)

    # [0,1] PPG时域
    axes[0, 1].plot(t_p, norm(ppg), color='#e74c3c', lw=0.8, alpha=0.9)
    axes[0, 1].set_title('参考PPG  时域波形（归一化）')
    axes[0, 1].set_xlabel('时间 (s)')
    axes[0, 1].set_ylabel('归一化幅値')
    axes[0, 1].grid(alpha=0.25)

    # [1,0] 座垫频域
    axes[1, 0].fill_between(fr_c[mc], ps_cn, alpha=0.2, color='#2980b9')
    axes[1, 0].plot(fr_c[mc], ps_cn, color='#2980b9', lw=1.5, label='座垫 PSD')
    axes[1, 0].axvline(peak_c, color='orange', lw=2, ls='--',
                       label=f'主峰 {peak_c:.1f} BPM')
    axes[1, 0].axvline(ref_hb, color='#e74c3c', lw=1.5, ls=':',
                       label=f'PPG参考 {ref_hb:.1f} BPM')
    axes[1, 0].set_title('座垫压力  功率谱 (30–150 BPM)')
    axes[1, 0].set_xlabel('BPM')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.25)

    # [1,1] PPG频域
    axes[1, 1].fill_between(fr_p[mp], ps_pn, alpha=0.2, color='#e74c3c')
    axes[1, 1].plot(fr_p[mp], ps_pn, color='#e74c3c', lw=1.5, label='PPG PSD')
    axes[1, 1].axvline(ref_hb, color='orange', lw=2, ls='--',
                       label=f'主峰 {ref_hb:.1f} BPM')
    axes[1, 1].axvline(peak_c, color='#2980b9', lw=1.5, ls=':',
                       label=f'座垫主峰 {peak_c:.1f} BPM')
    axes[1, 1].set_title('参考PPG  功率谱 (30–150 BPM)')
    axes[1, 1].set_xlabel('BPM')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.25)

    subj_dir = os.path.join(OUT_DIR, subj)
    os.makedirs(subj_dir, exist_ok=True)
    out_path = os.path.join(subj_dir, f'{subj}_预处理对比.png')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  {subj} -> {subj}/{os.path.basename(out_path)}')

# ── 8受试者汇总频谱图 ───────────────────────────────────────────
print('\n绘制汇总图...')
fig, axes = plt.subplots(4, 2, figsize=(16, 20), constrained_layout=True)
fig.suptitle('8受试者  全预处理后心跳功率谱汇总\n'
             '蓝色=座垫PSD  橙虚=座垫主峰  红点=PPG参考心率',
             fontsize=12, fontweight='bold')

for i, subj in enumerate(SUBJECTS):
    ax = axes[i // 2, i % 2]
    r = results[subj]
    fused = r['fused']; ref_hb = r['ref_hb']; peak_c = r['peak_c']
    err = abs(peak_c - ref_hb)
    flag = 'OK' if err <= 5 else ('~' if err <= 10 else 'X')

    n_c  = len(fused)
    fr_c = np.fft.rfftfreq(n_c, 1.0/FS)*60
    ps_c = np.abs(np.fft.rfft(fused - fused.mean()))**2
    mc   = (fr_c >= 30) & (fr_c <= 150)
    ps_cn = ps_c[mc] / ps_c[mc].max()

    ax.fill_between(fr_c[mc], ps_cn, alpha=0.2, color='#2980b9')
    ax.plot(fr_c[mc], ps_cn, color='#2980b9', lw=1.5)
    ax.axvline(peak_c, color='orange', lw=2, ls='--',
               label=f'座垫 {peak_c:.1f}')
    ax.axvline(ref_hb, color='#e74c3c', lw=1.5, ls=':',
               label=f'PPG {ref_hb:.1f}')
    ax.set_title(f'[{subj}]  误差={err:.1f} BPM  {flag}', fontsize=10)
    ax.set_xlabel('BPM')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

summary_path = os.path.join(OUT_DIR, '全受试者频谱汇总.png')
fig.savefig(summary_path, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'\n汇总图 -> {summary_path}')
print('\n完成！')
