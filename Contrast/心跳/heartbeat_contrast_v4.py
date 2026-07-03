# -*- coding: utf-8 -*-
"""
心跳提取对比 v4.0 — 频域谱减法 + 多窗口BPM集成
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v3 教训: 时域陷波器带宽太大, 会破坏与谐波相邻的心跳频率

v4 核心改进:
  改进H: 频域谱减法 (Spectral Subtraction)
         估计呼吸谐波的频域能量, 从信号PSD中减去,
         再在残差PSD中找心跳主频
         → 完全在频域操作, 不破坏时域信号

  改进I: 多窗口BPM集成
         用20s/25s/30s三种窗口分别计算BPM再取中位数
         提高在不同信号段的稳健性

  改进J: 长窗自相关 (使用全段信号做ACR)
         全段 (~80s) 做一次ACR, 频率分辨率更高
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
from scipy.signal import butter, filtfilt, find_peaks
from scipy.signal import correlate

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
STEP_SEC = 5.0
DEADZONE = 30
CLIP_MAX = 2000
HB_LOW, HB_HIGH = 0.8, 2.2
BPM_MIN, BPM_MAX = 40.0, 150.0

DATA_FILE = os.path.join(ROOT, 'data',         '20260702_160410_40x40.txt')
REF_FILE  = os.path.join(ROOT, 'Precise_Data', '刘若红0702.txt')
OUT_DIR   = os.path.join(ROOT, 'Contrast', '心跳', '刘若红_0702_160410_心跳对比_v4')
ALGO_NAMES = ['均值法', 'ACMD', 'VMD', 'EMD', 'VME']

def _font():
    for n in ['SimHei','Microsoft YaHei','WenQuanYi Micro Hei']:
        try:
            fm.findfont(fm.FontProperties(family=n), fallback_to_default=False)
            plt.rcParams.update({'font.family':n,'axes.unicode_minus':False})
            return
        except: pass
    plt.rcParams['axes.unicode_minus'] = False
_font()


# ════════════════════════════════════════════════════════════════
# 改进H: 频域谱减法 BPM 估计
# ════════════════════════════════════════════════════════════════
def bpm_spectral_sub(sig: np.ndarray, fs: float,
                     breath_freq: float) -> float:
    """
    1. 计算信号 PSD
    2. 估计呼吸谐波 PSD 贡献（高斯模型）并减去
    3. 在残差 PSD 的心跳频段找主峰
    完全在频域操作，不破坏时域信号。
    """
    n     = len(sig)
    freqs = np.fft.rfftfreq(n, 1.0/fs)
    psd   = np.abs(np.fft.rfft(sig - sig.mean()))**2

    # 估计并减去呼吸谐波模型
    psd_res = psd.copy().astype(np.float64)
    BW      = 0.04    # 每个谐波高斯半宽
    for k in range(1, 10):
        f_harm = breath_freq * k
        if f_harm >= fs / 2:
            break
        # 该谐波的局部能量（用于估计幅度）
        mask_h = np.abs(freqs - f_harm) <= BW
        if mask_h.any():
            peak_e = np.max(psd[mask_h])
            gauss  = peak_e * np.exp(-0.5 * ((freqs - f_harm) / (BW/2))**2)
            psd_res = np.maximum(0, psd_res - gauss)

    # 在心跳频段找主峰
    mask_hb = (freqs >= BPM_MIN/60) & (freqs <= BPM_MAX/60)
    if not mask_hb.any():
        return 0.0
    dom_f = freqs[mask_hb][np.argmax(psd_res[mask_hb])]
    bpm   = dom_f * 60
    return float(bpm) if BPM_MIN <= bpm <= BPM_MAX else 0.0


# ════════════════════════════════════════════════════════════════
# 改进I+J: 多窗口 + 长窗ACR BPM 集成
# ════════════════════════════════════════════════════════════════
def bpm_acr_long(sig: np.ndarray, fs: float) -> float:
    """全段信号做长窗自相关，频率分辨率更高"""
    n = len(sig)
    if n < 50: return 0.0
    s   = sig - sig.mean()
    acf = correlate(s, s, mode='full')[n-1:]
    acf = acf / (acf[0] + 1e-12)
    lg_min = max(1, int(60.0/BPM_MAX * fs))
    lg_max = min(n-1, int(60.0/BPM_MIN * fs))
    if lg_min >= lg_max: return 0.0
    seg = acf[lg_min:lg_max]
    pks, pr = find_peaks(seg, prominence=0.05)
    pk = (lg_min + int(pks[np.argmax(pr['prominences'])])
          if len(pks) else lg_min + int(np.argmax(seg)))
    bpm = 60.0 / (pk / fs)
    return float(bpm) if BPM_MIN <= bpm <= BPM_MAX else 0.0


def multiwin_bpm(sig: np.ndarray, fs: float,
                 breath_freq: float) -> tuple:
    """
    三种窗口(20/25/30s)的谱减法BPM + 全段ACR → 四值中位数
    返回 (vote, ss_list, acr_long)
    """
    ss_vals = []
    for win_s in [20.0, 25.0, 30.0]:
        win = int(win_s * fs)
        step = max(1, int(5.0 * fs))
        segs = []
        i = 0
        while i + win <= len(sig):
            segs.append(bpm_spectral_sub(sig[i:i+win], fs, breath_freq))
            i += step
        if segs:
            # 取中位数作为该窗口长度的BPM估计
            valid = [v for v in segs if BPM_MIN <= v <= BPM_MAX]
            ss_vals.append(float(np.median(valid)) if valid else 0.0)

    acr = bpm_acr_long(sig, fs)
    all_vals = [v for v in ss_vals + [acr] if BPM_MIN <= v <= BPM_MAX]
    vote = float(np.median(all_vals)) if all_vals else 0.0
    return vote, ss_vals, acr


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
            try: datetime.strptime(p[0],'%H:%M:%S.%f')
            except ValueError: continue
            raw = np.array(p[1:1601],dtype=np.float32).reshape(40,40)
            f   = np.clip(raw,0,CLIP_MAX).astype(np.float32)
            f[f<DEADZONE]=0; f=median_filter(f,size=3); f=gaussian_filter(f,sigma=0.5)
            frames.append(f)
    frames = np.array(frames,dtype=np.float32)
    trim   = int(TRIM_SEC*FS)
    if len(frames)>2*trim: frames=frames[trim:-trim]
    print(f"       {len(frames)} 帧 ({len(frames)/FS:.1f}s)")
    return frames


def load_ref_ch2(fp):
    print(f"[REF]  {os.path.basename(fp)}")
    fs_r = 2000.0
    with open(fp,'r',encoding='utf-8',errors='replace') as fh:
        lines = fh.readlines()
    for ln in lines[:4]:
        if 'msec/sample' in ln:
            try: fs_r=1000.0/float(ln.strip().split()[0])
            except: pass
            break
    di=0
    for i,ln in enumerate(lines):
        if ln.strip().startswith('CH1'): di=i+2; break

    ch1r, ch2r = [], []
    for ln in lines[di:]:
        cols=ln.strip().split('\t')
        try: ch1r.append(float(cols[0])); ch2r.append(float(cols[1]))
        except: continue

    # 呼吸基频
    ch1=np.array(ch1r,dtype=np.float64)
    trim=int(TRIM_SEC*fs_r)
    if len(ch1)>2*trim: ch1=ch1[trim:-trim]
    ch1-=ch1.mean()
    b,a=butter(4,1.0/(0.5*fs_r),btype='low')
    ch1_ds=filtfilt(b,a,ch1)[::max(1,int(fs_r/10))]
    fs_rsp=fs_r/max(1,int(fs_r/10))
    ch1_bp=butter_bandpass_filter(ch1_ds,0.1,0.5,fs=fs_rsp,order=4)
    fr=np.fft.rfftfreq(len(ch1_bp),1.0/fs_rsp)
    ps=np.abs(np.fft.rfft(ch1_bp))**2
    rm=(fr>=0.1)&(fr<=0.5)
    breath_freq=float(fr[rm][np.argmax(ps[rm])]) if rm.any() else 0.238
    print(f"       呼吸基频={breath_freq:.3f}Hz ({breath_freq*60:.1f}BPM) "
          f"→ 6次谐波={breath_freq*6*60:.1f}BPM")

    # 参考心率
    ch2=np.array(ch2r,dtype=np.float64)
    if len(ch2)>2*trim: ch2=ch2[trim:-trim]
    ch2-=ch2.mean()
    b2,a2=butter(4,5.0/(0.5*fs_r),btype='low')
    ds=max(1,int(fs_r/50))
    ch2_ds=filtfilt(b2,a2,ch2)[::ds]; fs_ppg=fs_r/ds
    ch2_bp=butter_bandpass_filter(ch2_ds,HB_LOW,HB_HIGH,fs=fs_ppg,order=4)
    vote,ss_v,acr=multiwin_bpm(ch2_bp,fs_ppg,breath_freq)
    print(f"       参考心率: 谱减={ss_v}  ACR长窗={acr:.1f}  投票={vote:.1f}")
    return ch2_bp,float(fs_ppg),float(vote),float(breath_freq)


# ════════════════════════════════════════════════════════════════
# ROI + ICA + 轻量VME单次基线去除 + 带通
# ════════════════════════════════════════════════════════════════
def _split_col(mf):
    return 12+int(np.argmin(mf.sum(axis=0)[12:28]))

def _pick_centers(zone,k,md,c_off):
    order=np.argsort(zone.ravel())[::-1]; cens=[]
    for idx in order:
        r,cl=np.unravel_index(idx,zone.shape); c=cl+c_off
        if not any(max(abs(r-cr),abs(c-cc))<md for cr,cc in cens):
            cens.append((r,c))
        if len(cens)==k: break
    while len(cens)<k: cens.append((zone.shape[0]//2,c_off+zone.shape[1]//2))
    return cens

def _ica_cardiac(roi_mat,fs):
    from sklearn.decomposition import FastICA,PCA
    from scipy.fft import fft as _fft,fftfreq as _ff
    M,N=roi_mat.shape
    if M==1: return roi_mat[0]
    X=roi_mat.T; nc=min(M,5)
    try:
        src=FastICA(n_components=nc,random_state=42,max_iter=2000,tol=1e-3).fit_transform(X)
    except: src=PCA(n_components=nc,random_state=42).fit_transform(X)
    freqs=_ff(N,1.0/fs)[:N//2]; inb=(freqs>=HB_LOW)&(freqs<=HB_HIGH)
    best,bsnr=None,-999.0
    for k in range(src.shape[1]):
        c=src[:,k]; psd=np.abs(_fft(c))[:N//2]**2
        snr=10*np.log10(psd[inb].sum()/(psd[~inb&(freqs>0)].sum()+1e-9))
        if snr>bsnr: bsnr,best=snr,c
    ms=roi_mat.mean(axis=0)
    return best if np.dot(best,ms)>=0 else -best

def build_fused_signal(frames,breath_freq):
    mf=frames.mean(axis=0); sp=_split_col(mf)
    lc=_pick_centers(mf[:,:sp],K_ROIS,MIN_DIST,0)
    rc=_pick_centers(mf[:,sp:],K_ROIS,MIN_DIST,sp)
    rois=([{'label':f'L{i+1}','c':c} for i,c in enumerate(lc)]+
          [{'label':f'R{i+1}','c':c} for i,c in enumerate(rc)])
    half=ROI_SIZE//2; H,W=frames.shape[1],frames.shape[2]
    sigs=[]
    for roi in rois:
        r,c=roi['c']
        rs,re=max(0,r-half),min(H,r+half+1)
        cs,ce=max(0,c-half),min(W,c+half+1)
        ts=frames[:,rs:re,cs:ce].mean(axis=(1,2))
        t2=np.arange(len(ts),dtype=np.float64)
        ts=ts-np.polyval(np.polyfit(t2,ts,3),t2)
        ts=wavelet_denoise(ts,alpha=0.3)
        sigs.append(ts.astype(np.float64))
        print(f"       {roi['label']}: 行{r:2d}列{c:2d}  压力={mf[r,c]:.1f}")
    roi_mat=np.array(sigs); fused=_ica_cardiac(roi_mat,FS)

    # 单次VME去呼吸基线（不去谐波）
    from algorithms.base import VME_Core
    try:
        u=VME_Core(fused-fused.mean(),fs=FS,f_init=breath_freq,alpha=1000)
        fused=fused-u
    except: pass

    fused=butter_bandpass_filter(fused,HB_LOW,HB_HIGH,fs=FS,order=4)
    print(f"       ICA+单VME 融合信号 shape={fused.shape}")
    return fused,rois,mf


# ════════════════════════════════════════════════════════════════
# 算法执行
# ════════════════════════════════════════════════════════════════
def run_all(fused, frames, breath_freq):
    algo_map = {
        '均值法': lambda s,f: extract_heartbeat_mean(s, fs=FS),
        'ACMD':   lambda s,f: extract_heartbeat_acmd(s, fs=FS),
        'VMD':    lambda s,f: extract_heartbeat_vmd (s, fs=FS),
        'EMD':    lambda s,f: extract_heartbeat_emd (s, fs=FS),
        'VME':    lambda s,f: extract_heartbeat_vme (s, fs=FS),
    }
    print('\n[ALGO]  (谱减法多窗口+长窗ACR投票)')
    results = {}
    for name in ALGO_NAMES:
        print(f"    {name}...", end='', flush=True)
        t0 = time.perf_counter()
        try:
            sig = np.array(algo_map[name](fused, frames),
                           dtype=np.float64).flatten()
            if len(sig) != len(fused):
                sig = (sig[:len(fused)] if len(sig) > len(fused)
                       else np.pad(sig, (0, len(fused)-len(sig))))
            vote, ss_v, acr = multiwin_bpm(sig, FS, breath_freq)
        except Exception as e:
            print(f" [err:{e}]")
            sig = np.zeros_like(fused); vote = acr = 0.0; ss_v = []
        el = (time.perf_counter()-t0)*1000
        results[name] = {
            'sig': sig, 'bpm': vote,
            'ss': float(np.median([v for v in ss_v if v > 0]) if ss_v else 0),
            'acr': float(acr), 'time_ms': float(el),
        }
        print(f"  谱减中位={results[name]['ss']:.1f}  "
              f"ACR={acr:.1f}  投票={vote:.1f}  耗时={el:.0f}ms")
    return results


# ════════════════════════════════════════════════════════════════
# 绘图 + CSV + main()
# ════════════════════════════════════════════════════════════════
def sw_bpm_ss(sig, fs, bf):
    win, step = int(25.0*fs), int(STEP_SEC*fs)
    T, B = [], []
    i = 0
    while i+win <= len(sig):
        T.append((i+win/2)/fs)
        B.append(bpm_spectral_sub(sig[i:i+win], fs, bf))
        i += step
    return np.array(T), np.array(B)


def plot_one(name, res, ref_sig, ref_fs, ref_bpm, bf, out_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7),
                                    constrained_layout=True)
    err = abs(res['bpm']-ref_bpm)
    fig.suptitle(
        f'[{name}]  投票={res["bpm"]:.1f} BPM  |  参考={ref_bpm:.1f}  |  '
        f'误差={err:.1f} BPM  |  耗时={res["time_ms"]:.0f}ms',
        fontsize=11, fontweight='bold')

    rT, rB = sw_bpm_ss(ref_sig, ref_fs, bf)
    aT, aB = sw_bpm_ss(res['sig'], FS, bf)
    ax1.plot(rT, rB, 'r-', lw=2.2, label=f'参考PPG ({ref_bpm:.1f})')
    ax1.plot(aT, aB, color='#27ae60', lw=1.8, label=f'{name} ({res["bpm"]:.1f})')
    ax1.axhline(ref_bpm, color='red', lw=0.8, ls='--', alpha=0.4)
    ax1.set_ylabel('BPM'); ax1.set_xlabel('时间(s)')
    ax1.set_title('滑动窗口BPM (25s窗口, 谱减法)')
    ax1.legend(fontsize=10); ax1.grid(alpha=0.25)
    yc = ref_bpm if ref_bpm > 0 else 75
    ax1.set_ylim(max(0, yc-30), yc+30)

    def _n(s): sd=np.std(s); return s/sd if sd>1e-9 else s
    ax2.plot(np.arange(len(ref_sig))/ref_fs, _n(ref_sig),
             'r-', lw=1.5, alpha=0.8, label='参考PPG')
    ax2.plot(np.arange(len(res['sig']))/FS, _n(res['sig']),
             color='#2980b9', lw=1.2, alpha=0.8, label=f'ICA+{name}')
    ax2.set_ylabel('归一化幅值'); ax2.set_xlabel('时间(s)')
    ax2.set_title('提取信号波形')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.25); ax2.set_ylim(-5, 5)

    safe = name.replace(' ','_').replace('-','_')
    path = os.path.join(out_dir, f'{safe}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {name:<12} → {os.path.basename(path)}")


def plot_summary(all_res, ref_bpm, out_dir):
    names = list(all_res.keys())
    errs  = [abs(all_res[n]['bpm']-ref_bpm) for n in names]
    times = [all_res[n]['time_ms'] for n in names]
    fig, ax1 = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax2 = ax1.twinx(); x, w = np.arange(len(names)), 0.35
    c_e = ['#27ae60' if e<=5 else '#f39c12' if e<=10 else '#e74c3c' for e in errs]
    b1 = ax1.bar(x-w/2, errs,  w, color=c_e,      alpha=0.88, label='误差(左)')
    b2 = ax2.bar(x+w/2, times, w, color='#5b9bd5', alpha=0.72, label='耗时ms(右)')
    for bar,v in zip(b1,errs):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    for bar,v in zip(b2,times):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f'{v:.0f}', ha='center', va='bottom', fontsize=8, color='#2e6da4')
    ax1.axhline(5,  color='green',  lw=1.2, ls='--', alpha=0.7, label='±5 BPM')
    ax1.axhline(10, color='orange', lw=1.2, ls='--', alpha=0.7, label='±10 BPM')
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=11)
    ax1.set_ylabel('|BPM误差|'); ax2.set_ylabel('耗时(ms)', color='#2e6da4')
    ax1.set_title(f'心跳算法 v4 (参考={ref_bpm:.1f} BPM, ICA+单VME+谱减法)')
    l1,lb1=ax1.get_legend_handles_labels(); l2,lb2=ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lb1+lb2, fontsize=8, loc='upper left')
    ax1.grid(alpha=0.25, axis='y')
    path = os.path.join(out_dir, '汇总误差与耗时.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  汇总图 → {path}")


def save_csv(all_res, ref_bpm, out_dir):
    path = os.path.join(out_dir, '心跳结果汇总.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['算法','投票BPM','谱减中位','ACR长窗','参考BPM',
                    '绝对误差','相对误差(%)','耗时(ms)'])
        for nm, r in all_res.items():
            err = abs(r['bpm']-ref_bpm)
            rel = err/ref_bpm*100 if ref_bpm > 0 else float('nan')
            w.writerow([nm,f"{r['bpm']:.2f}",f"{r['ss']:.2f}",
                        f"{r['acr']:.2f}",f"{ref_bpm:.2f}",
                        f"{err:.2f}",f"{rel:.1f}",f"{r['time_ms']:.0f}"])
    print(f"  CSV → {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print('\n'+'='*64)
    print('  心跳提取对比 v4.0  (ICA+单VME+谱减法多窗口+长窗ACR)')
    print(f'  输出: {OUT_DIR}')
    print('='*64)

    frames = load_cushion(DATA_FILE)
    ref_sig, ref_fs, ref_bpm, breath_freq = load_ref_ch2(REF_FILE)

    print('\n[预处理]')
    t0 = time.perf_counter()
    fused, rois, mf = build_fused_signal(frames, breath_freq)
    print(f"       耗时: {(time.perf_counter()-t0)*1000:.0f}ms")

    all_res = run_all(fused, frames, breath_freq)

    print('\n[PLOT]')
    for name in ALGO_NAMES:
        plot_one(name, all_res[name], ref_sig, ref_fs, ref_bpm, breath_freq, OUT_DIR)
    plot_summary(all_res, ref_bpm, OUT_DIR)

    print('\n[CSV]')
    save_csv(all_res, ref_bpm, OUT_DIR)

    print('\n'+'='*64)
    print(f'  参考={ref_bpm:.1f} BPM  |  呼吸={breath_freq*60:.1f} BPM  |  '
          f'6次谐波={breath_freq*6*60:.1f} BPM')
    print(f'  {"算法":<12}  {"投票":>7}  {"谱减":>7}  {"ACR":>7}  {"误差":>7}  {"耗时":>7}')
    print(f'  {"-"*54}')
    for nm in ALGO_NAMES:
        r = all_res[nm]; err = abs(r['bpm']-ref_bpm)
        flg = 'OK' if err<=5 else ('~' if err<=10 else 'X')
        print(f'  {nm:<12}  {r["bpm"]:>7.1f}  {r["ss"]:>7.1f}  '
              f'{r["acr"]:>7.1f}  {err:>7.1f}  {r["time_ms"]:>7.0f}  {flg}')
    print('='*64+'\n')
    print(f'完成 → {OUT_DIR}\n')


if __name__ == '__main__':
    main()
