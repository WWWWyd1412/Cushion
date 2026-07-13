# -*- coding: utf-8 -*-
"""
raw_hb_plot.py
对8个受试者的40×40座垫数据，经0.75-2.5Hz带通滤波后，
绘制每个受试者的时域波形 + 频域PSD，以及参考PPG对比。
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), '40_40_Extraction_1'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime
from scipy.ndimage import median_filter, gaussian_filter
from scipy.signal import butter, filtfilt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSHION_DIR = os.path.join(ROOT, '40_40_Cushion_Data')
PPG_DIR     = os.path.join(ROOT, 'PPGdataset')
OUT_DIR     = os.path.join(ROOT, 'Contrast', '心跳')
os.makedirs(OUT_DIR, exist_ok=True)

FS       = 11.18
TRIM_SEC = 20.0
F_LOW    = 0.75
F_HIGH   = 2.5
SUBJECTS = ['lbx1','lbx2','wyd1','wyd2','xxr1','xxr2','zxc1','zxc2']

# ── 字体 ─────────────────────────────────────────────────────────
def _font():
    for n in ['SimHei','Microsoft YaHei','WenQuanYi Micro Hei']:
        try:
            fm.findfont(fm.FontProperties(family=n), fallback_to_default=False)
            plt.rcParams.update({'font.family': n, 'axes.unicode_minus': False})
            return
        except: pass
    plt.rcParams['axes.unicode_minus'] = False
_font()

# ── 带通滤波 ─────────────────────────────────────────────────────
def bandpass(sig, fs, lo, hi, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, sig)

# ── 加载座垫 ─────────────────────────────────────────────────────
def load_cushion(fp):
    frames = []
    with open(fp, 'r', encoding='utf-8') as fh:
        for line in fh:
            p = line.split()
            if len(p) < 1601: continue
            try: datetime.strptime(p[0], '%H:%M:%S.%f')
            except: continue
            raw = np.array(p[1:1601], dtype=np.float32).reshape(40,40)
            f = raw.astype(np.float32)
            if f.mean() > 1000: f = 4095 - f
            f = np.clip(f, 0, 1200); f[f<30] = 0
            f = median_filter(f, size=3); f = gaussian_filter(f, sigma=0.5)
            frames.append(f)
    frames = np.array(frames, dtype=np.float32)
    trim = int(TRIM_SEC * FS)
    return frames[trim:-trim] if len(frames) > 2*trim else frames

# ── 加载参考PPG ──────────────────────────────────────────────────
def load_ppg(fp, fs_ppg=50.0):
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
    ch2 = []
    for ln in lines[di:]:
        cols = ln.strip().split('\t')
        try: ch2.append(float(cols[1]))
        except: continue
    ch2 = np.array(ch2, dtype=np.float64)
    trim = int(TRIM_SEC * fs_r)
    if len(ch2) > 2*trim: ch2 = ch2[trim:-trim]
    ch2 -= ch2.mean()
    b, a  = butter(4, 5.0/(0.5*fs_r), btype='low')
    lp    = filtfilt(b, a, ch2)
    ds    = max(1, int(fs_r / fs_ppg))
    sig   = lp[::ds]; fs_act = fs_r/ds
    sig   = bandpass(sig, fs_act, F_LOW, F_HIGH)
    return sig, fs_act

# ── 绘图主函数 ───────────────────────────────────────────────────
def plot_subject(subject, ref_hb):
    cushion_fp = os.path.join(CUSHION_DIR, f'{subject}.txt')
    ppg_fp     = os.path.join(PPG_DIR,     f'{subject}.txt')

    # 加载数据
    frames = load_cushion(cushion_fp)
    N = len(frames)
    t_cush = np.arange(N) / FS

    # 取全局均值信号
    raw_sig = frames.mean(axis=(1,2))
    t2 = np.arange(N, dtype=np.float64)
    raw_sig -= np.polyval(np.polyfit(t2, raw_sig, 3), t2)   # poly detrend

    # 带通滤波
    filt_sig = bandpass(raw_sig, FS, F_LOW, F_HIGH)

    # 参考PPG
    ppg_sig, fs_ppg = load_ppg(ppg_fp)
    t_ppg = np.arange(len(ppg_sig)) / fs_ppg

    # 频域 PSD
    def get_psd(sig, fs):
        n  = len(sig)
        fr = np.fft.rfftfreq(n, 1/fs)
        ps = np.abs(np.fft.rfft(sig - sig.mean()))**2
        ps = ps / ps.max()   # 归一化
        return fr, ps

    fr_c, ps_c = get_psd(filt_sig, FS)
    fr_p, ps_p = get_psd(ppg_sig,  fs_ppg)

    # FFT主频 → BPM
    m_c = (fr_c >= F_LOW)  & (fr_c <= F_HIGH)
    m_p = (fr_p >= F_LOW)  & (fr_p <= F_HIGH)
    dom_c = fr_c[m_c][np.argmax(ps_c[m_c])] * 60 if m_c.any() else 0
    dom_p = fr_p[m_p][np.argmax(ps_p[m_p])] * 60 if m_p.any() else 0

    # ── 绘图 ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(
        f'[{subject}]  0.75–2.5 Hz 带通滤波\n'
        f'座垫FFT主频={dom_c:.1f} BPM  |  参考PPG心率={ref_hb:.1f} BPM',
        fontsize=13, fontweight='bold'
    )

    # ── 时域：座垫 ──
    ax = axes[0, 0]
    ax.plot(t_cush, filt_sig, color='#2980b9', lw=1.0, alpha=0.9)
    ax.set_title('座垫压力信号（时域，0.75–2.5 Hz带通）')
    ax.set_xlabel('时间 (s)');  ax.set_ylabel('幅值（任意单位）')
    ax.grid(alpha=0.25)

    # ── 时域：PPG ──
    ax = axes[0, 1]
    ax.plot(t_ppg, ppg_sig, color='#e74c3c', lw=1.0, alpha=0.9)
    ax.set_title(f'参考PPG（时域，0.75–2.5 Hz带通）')
    ax.set_xlabel('时间 (s)');  ax.set_ylabel('幅值 (V)')
    ax.grid(alpha=0.25)

    # ── 频域：座垫 ──
    ax = axes[1, 0]
    m = (fr_c >= 0.5) & (fr_c <= 3.0)
    ax.plot(fr_c[m]*60, ps_c[m], color='#2980b9', lw=1.5)
    ax.axvline(dom_c, color='#2980b9', ls='--', lw=1.2,
               label=f'主频 {dom_c:.1f} BPM')
    ax.axvline(ref_hb, color='#e74c3c', ls='--', lw=1.2,
               label=f'真实 {ref_hb:.1f} BPM')
    # 标注呼吸谐波位置（灰色）
    ax.set_title('座垫压力信号 PSD（归一化）')
    ax.set_xlabel('频率 (BPM)');  ax.set_ylabel('归一化功率')
    ax.set_xlim(30, 160)
    ax.legend(fontsize=9);  ax.grid(alpha=0.25)

    # ── 频域：PPG ──
    ax = axes[1, 1]
    m = (fr_p >= 0.5) & (fr_p <= 3.0)
    ax.plot(fr_p[m]*60, ps_p[m], color='#e74c3c', lw=1.5)
    ax.axvline(dom_p, color='#e74c3c', ls='--', lw=1.2,
               label=f'主频 {dom_p:.1f} BPM')
    ax.axvline(ref_hb, color='#2980b9', ls=':', lw=1.2,
               label=f'真实 {ref_hb:.1f} BPM')
    ax.set_title('参考PPG PSD（归一化）')
    ax.set_xlabel('频率 (BPM)');  ax.set_ylabel('归一化功率')
    ax.set_xlim(30, 160)
    ax.legend(fontsize=9);  ax.grid(alpha=0.25)

    out_path = os.path.join(OUT_DIR, f'{subject}_原始心跳频谱.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [{subject}] 座垫={dom_c:.1f} BPM  PPG={dom_p:.1f} BPM  '
          f'真实={ref_hb:.1f} BPM  → {os.path.basename(out_path)}')


# ── 汇总对比图 ───────────────────────────────────────────────────
def plot_all_subjects(subject_hb):
    n = len(subject_hb)
    fig, axes = plt.subplots(n, 2, figsize=(16, n*3.2), constrained_layout=True)
    fig.suptitle('所有受试者座垫压力 vs 参考PPG  —  0.75–2.5 Hz带通', fontsize=13)

    for i, (subject, ref_hb) in enumerate(subject_hb):
        cushion_fp = os.path.join(CUSHION_DIR, f'{subject}.txt')
        ppg_fp     = os.path.join(PPG_DIR,     f'{subject}.txt')
        frames = load_cushion(cushion_fp)
        N = len(frames); t2 = np.arange(N, dtype=np.float64)
        raw_sig = frames.mean(axis=(1,2))
        raw_sig -= np.polyval(np.polyfit(t2, raw_sig, 3), t2)
        filt_sig = bandpass(raw_sig, FS, F_LOW, F_HIGH)

        ppg_sig, fs_ppg = load_ppg(ppg_fp)

        def get_psd(sig, fs):
            n  = len(sig); fr = np.fft.rfftfreq(n, 1/fs)
            ps = np.abs(np.fft.rfft(sig-sig.mean()))**2
            return fr, ps/ps.max()

        fr_c, ps_c = get_psd(filt_sig, FS)
        fr_p, ps_p = get_psd(ppg_sig,  fs_ppg)

        m_c = (fr_c >= F_LOW) & (fr_c <= F_HIGH)
        m_p = (fr_p >= F_LOW) & (fr_p <= F_HIGH)
        dom_c = fr_c[m_c][np.argmax(ps_c[m_c])]*60 if m_c.any() else 0
        dom_p = fr_p[m_p][np.argmax(ps_p[m_p])]*60 if m_p.any() else 0

        # 座垫频域
        ax = axes[i, 0]
        m = (fr_c >= 0.5) & (fr_c <= 3.0)
        ax.fill_between(fr_c[m]*60, ps_c[m], alpha=0.35, color='#2980b9')
        ax.plot(fr_c[m]*60, ps_c[m], color='#2980b9', lw=1.3)
        ax.axvline(ref_hb,  color='red',      lw=1.5, ls='--', label=f'真实 {ref_hb:.0f}')
        ax.axvline(dom_c,   color='#2980b9',  lw=1.2, ls=':',  label=f'主频 {dom_c:.0f}')
        ax.set_title(f'[{subject}] 座垫 PSD  (真实={ref_hb:.0f} BPM)', fontsize=9)
        ax.set_xlim(30, 160); ax.set_ylabel('功率(归一)'); ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc='upper right')

        # PPG频域
        ax = axes[i, 1]
        m2 = (fr_p >= 0.5) & (fr_p <= 3.0)
        ax.fill_between(fr_p[m2]*60, ps_p[m2], alpha=0.35, color='#e74c3c')
        ax.plot(fr_p[m2]*60, ps_p[m2], color='#e74c3c', lw=1.3)
        ax.axvline(ref_hb, color='red',     lw=1.5, ls='--', label=f'真实 {ref_hb:.0f}')
        ax.axvline(dom_p,  color='#e74c3c', lw=1.2, ls=':',  label=f'主频 {dom_p:.0f}')
        ax.set_title(f'[{subject}] 参考PPG PSD', fontsize=9)
        ax.set_xlim(30, 160); ax.set_ylabel('功率(归一)'); ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc='upper right')

    axes[-1, 0].set_xlabel('BPM')
    axes[-1, 1].set_xlabel('BPM')
    out = os.path.join(OUT_DIR, '全受试者_心跳频谱对比.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'汇总图 → {out}')


# ── 主入口 ───────────────────────────────────────────────────────
REF_HB = {'lbx1':68.2,'lbx2':68.2,'wyd1':54.5,'wyd2':55.6,
          'xxr1':56.6,'xxr2':57.7,'zxc1':53.6,'zxc2':61.2}

if __name__ == '__main__':
    print('═'*60)
    print(f'  座垫压力 0.75–2.5 Hz 带通 时域+频域 可视化')
    print('═'*60)
    for s in SUBJECTS:
        plot_subject(s, REF_HB[s])
    plot_all_subjects([(s, REF_HB[s]) for s in SUBJECTS])
    print('完成！')
