import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, savgol_filter, welch, butter, filtfilt
import pywt


def butter_bandpass_filter(data, lowcut=0.8, highcut=2.2, fs=10.0, order=3):
    """【频域滤波】专门针对心搏信号（0.8 - 2.2 Hz）的带通巴特沃斯滤波器"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # 限制在有效 Nyquist 频率范围内
    if high >= 1.0:
        high = 0.99
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def wavelet_denoise(signal, alpha=0.3):
    """【预处理】微弱信号自适应小波去噪：保留心搏（J峰）等微弱高频特征"""
    if len(signal) < 16:
        return signal
    # 使用 db4 小波进行 3 层分解
    coeffs = pywt.wavedec(signal, 'db4', level=3)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    if sigma < 1e-12:
        return signal
    thr = alpha * np.sqrt(2.0 * np.log(len(signal))) * sigma
    coeffs_dn = [coeffs[0]] + [pywt.threshold(c, thr, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs_dn, 'db4')[:len(signal)]


def calculate_snr(signal, fs=10.0, band=(0.8, 2.2)):
    """【质量评估】计算分量在心跳频段内的信噪比 (SNR)"""
    nperseg = min(len(signal), 256)
    if nperseg < 8:
        return -10.0
    f, psd = welch(signal, fs, nperseg=nperseg)
    idx_band = np.logical_and(f >= band[0], f <= band[1])
    if not np.any(idx_band):
        return -10.0
    signal_pwr = np.sum(psd[idx_band])
    noise_pwr = np.sum(psd[~idx_band])
    return 10 * np.log10(signal_pwr / noise_pwr) if noise_pwr > 0 else 20.0


def VME_Core(signal, fs, f_init=0.25, alpha=1000, tol=1e-6, max_iter=150):
    """
    自适应基线漂移去除的核心算法：变分模态提取 (VME)
    用于提取特定频率（呼吸/超低频漂移，约 0.25 Hz）的干扰分量并将其剔除。
    引入镜像延拓 (Mirror Padding) 解决短信号边界瞬态畸变问题。
    """
    T = len(signal)
    if T < 10:
        return np.zeros_like(signal)
    
    # 1. 镜像延拓 (Mirror Padding) 消除边界效应
    pad_len = min(T // 2, 100)
    left_pad = signal[1:pad_len+1][::-1]
    right_pad = signal[-pad_len-1:-1][::-1]
    padded_signal = np.concatenate((left_pad, signal, right_pad))
    T_pad = len(padded_signal)
    
    # 2. 构造解析信号的傅里叶变换
    f_fft = np.fft.fft(padded_signal)
    half_T = T_pad // 2
    f_fft_analytic = np.zeros_like(f_fft, dtype=complex)
    f_fft_analytic[0] = f_fft[0]
    f_fft_analytic[1:half_T] = 2.0 * f_fft[1:half_T]
    if T_pad % 2 == 0:
        f_fft_analytic[half_T] = f_fft[half_T]
        
    freqs = np.fft.fftfreq(T_pad, 1/fs)
    
    u_fft = np.zeros(T_pad, dtype=complex)
    lambda_fft = np.zeros(T_pad, dtype=complex)
    omega_d = f_init  # 初始中心频率 (Hz)
    tau = 0.1         # 更新参数
    
    for it in range(max_iter):
        u_fft_old = u_fft.copy()
        
        diff = freqs[:half_T+1] - omega_d
        diff2 = diff**2
        diff4 = diff2**2
        
        # 对应 2024 论文中的 Eq (7) 更新公式
        num = f_fft_analytic[:half_T+1] + (alpha**2) * diff4 * u_fft_old[:half_T+1] + lambda_fft[:half_T+1] / 2.0
        den = (1.0 + (alpha**2) * diff4) * (1.0 + 2.0 * alpha * diff2)
        
        u_fft[:half_T+1] = num / (den + 1e-12)
        u_fft[half_T+1:] = 0.0
        
        # 更新中心频率 (Eq 8)
        u_power = np.abs(u_fft[:half_T+1])**2
        sum_power = np.sum(u_power)
        if sum_power > 1e-12:
            omega_d = np.sum(freqs[:half_T+1] * u_power) / sum_power
            
        # 更新 Lagrange 乘子 (Eq 9)
        error = (f_fft_analytic[:half_T+1] - u_fft[:half_T+1]) / (1.0 + (alpha**2) * diff4 + 1e-12)
        lambda_fft[:half_T+1] = lambda_fft[:half_T+1] + tau * error
        
        # 收敛判定
        if it > 5:
            change = np.linalg.norm(u_fft[:half_T+1] - u_fft_old[:half_T+1]) / (np.linalg.norm(u_fft_old[:half_T+1]) + 1e-12)
            if change < tol:
                break
                
    u_time_padded = np.real(np.fft.ifft(u_fft))
    # 3. 截除镜像延拓部分，恢复原始长度
    u_time = u_time_padded[pad_len : pad_len + T]
    return u_time


def get_dual_roi_mean(frames, window_size=5):
    """
    【自适应稳定态心脉降维提取】
    通过定位人左/右臀部最大压力点，进行高频心搏降维提取，并加入基于 VME 的自适应基线漂移 (BD) 去除。
    """
    if len(frames) == 0:
        return np.array([])
    
    offset = window_size // 2
    signal_1d = []
    
    # 智能寻找真正的稳定受力画面
    stable_mean_frame = None
    trigger_threshold = 120   # 压力触发门限
    stability_window = 20    # 稳定窗口长度
    
    # print("[HEARTBEAT-PROCESS] 开始扫描自适应坐姿稳定区间...")
    
    for i in range(len(frames) - stability_window):
        if np.max(frames[i]) > trigger_threshold:
            sub_series = frames[i : i + stability_window]
            frame_means = [np.mean(f) for f in sub_series]
            stability_score = np.std(frame_means)
            
            if stability_score < 5.0:
                stable_mean_frame = np.mean(sub_series, axis=0)
                # print(f"【触发成功】在第 {i} 帧捕捉到稳定坐姿（稳定度: {stability_score:.2f}）！")
                break

    if stable_mean_frame is None:
        print("【警告】未检测到绝对稳定区间，将采用全段均值进行坐标定位。")
        stable_mean_frame = np.mean(frames, axis=0)

    # 依据稳定画面锁定 ROI 左右坐标
    l_zone = stable_mean_frame[:, :16]
    r_zone = stable_mean_frame[:, 16:]
    
    l_idx = np.unravel_index(np.argmax(l_zone), l_zone.shape)
    r_idx = np.unravel_index(np.argmax(r_zone), r_zone.shape)
    r_idx = (r_idx[0], r_idx[1] + 16)
    
    print(f"左臀中心: {l_idx}, 右臀中心: {r_idx}。")

    for f in frames:
        def get_roi_mean(matrix, center_idx):
            r, c = center_idx
            r_s, r_e = max(0, r - offset), min(31, r + offset + 1)
            c_s, c_e = max(0, c - offset), min(31, c + offset + 1)
            roi = matrix[r_s:r_e, c_s:c_e]
            return np.mean(roi) if roi.size > 0 else 0

        signal_1d.append((get_roi_mean(f, l_idx) + get_roi_mean(f, r_idx)) / 2)

    sig_raw_np = np.array(signal_1d)
    sig_demeaned = sig_raw_np - np.mean(sig_raw_np)

    # 1. 自动估计信号在 [0.8, 2.2] Hz 生理区间内的主要频率 f_heart
    n_len = len(sig_demeaned)
    if n_len > 8:
        fft_vals = np.abs(np.fft.fft(sig_demeaned))[:n_len // 2]
        freqs = np.fft.fftfreq(n_len, 1/10.0)[:n_len // 2]
        valid_mask = (freqs >= 0.8) & (freqs <= 2.2)
        if np.any(valid_mask):
            f_heart = freqs[valid_mask][np.argmax(fft_vals[valid_mask])]
        else:
            f_heart = 1.2
    else:
        f_heart = 1.2

    # 2. 计算自适应基线剥离平衡因子 alpha_bd (Eq 10)
    if f_heart <= 1.25:
        # 心搏频率接近呼吸时，增大 alpha_bd 以缩窄基线提取带宽，防止滤除心跳信号
        alpha_bd = 1000.0 * np.exp(1.09 * ((f_heart - 1.25) / -0.5) ** 2)
    else:
        alpha_bd = 1000.0

    # 3. 运行 VME 提取低频漂移/呼吸成分 (f_init = 0.25 Hz, 即 15 BPM 呼吸)
    u_BD = VME_Core(sig_demeaned, fs=10.0, f_init=0.25, alpha=alpha_bd)
    sig_bd_removed = sig_demeaned - u_BD

    # 4. 后续级联链路：缓和小波去噪 -> 带通滤波 (双重保障)
    sig_wavelet = wavelet_denoise(sig_bd_removed, alpha=0.3)
    sig_bandpass = butter_bandpass_filter(sig_wavelet, lowcut=0.8, highcut=2.2, fs=10.0, order=3)
    
    return sig_bandpass


def select_best_component(components, fs):
    """筛选 0.8 - 2.2 Hz 心跳频段内能量最大的分量"""
    if components is None or len(components) == 0:
        return np.zeros(200)

    best_comp = None
    max_energy = -1
    for comp in components:
        n = len(comp)
        if n == 0:
            continue

        freqs = fftfreq(n, 1 / fs)[:n // 2]
        fft_vals = np.abs(fft(comp))[:n // 2]
        dom_freq = freqs[np.argmax(fft_vals)]

        # 心跳频段判定 0.8 - 2.2 Hz
        if 0.8 <= dom_freq <= 2.2:
            energy = np.sqrt(np.mean(comp ** 2))
            if energy > max_energy:
                max_energy = energy
                best_comp = comp

    if best_comp is not None:
        return best_comp
    else:
        return np.zeros_like(components[0]) if len(components) > 0 else np.zeros(200)


def reconstruct_multicomponent_with_snr(components, fs, snr_threshold=3.0):
    """
    【心跳重构】多通道叠加：
    融合所有落入心搏生理频段 (0.8 - 2.2Hz) 且 SNR 达标的分量。
    """
    if components is None or len(components) == 0:
        return np.zeros(100)

    reconstructed_signal = np.zeros_like(components[0])
    found_any = False

    for comp in components:
        n = len(comp)
        freqs = fftfreq(n, 1/fs)[:n // 2]
        fft_vals = np.abs(fft(comp))[:n // 2]
        dom_freq = freqs[np.argmax(fft_vals)]

        if 0.8 <= dom_freq <= 2.2:
            snr = calculate_snr(comp, fs)
            if snr >= snr_threshold:
                reconstructed_signal += comp
                found_any = True

    return reconstructed_signal if found_any else components[0]


def reconstruct_top3_components_by_energy(components, fs):
    """
    【心搏自适应重构】Top 3 能量重构逻辑：
    筛选所有落入心搏频段的分量，并选取能量前3的模态进行线性叠加。
    """
    if components is None or len(components) == 0:
        return np.zeros(100)

    valid_components_info = []
    
    for idx, comp in enumerate(components):
        n = len(comp)
        freqs = fftfreq(n, 1/fs)[:n // 2]
        fft_vals = np.abs(fft(comp))[:n // 2]
        dom_freq = freqs[np.argmax(fft_vals)]

        if 0.8 <= dom_freq <= 2.2:
            rms_energy = np.sqrt(np.mean(comp ** 2))
            valid_components_info.append({
                'component': comp,
                'energy': rms_energy,
                'freq': dom_freq
            })

    if len(valid_components_info) == 0:
        print("[心跳重构警告] 未发现任何落入心跳频段的分量，返回第一分量")
        return components[0]

    valid_components_info.sort(key=lambda x: x['energy'], reverse=True)
    top_k = min(3, len(valid_components_info))
    selected_info = valid_components_info[:top_k]
    
    reconstructed_signal = np.zeros_like(components[0])
    for info in selected_info:
        reconstructed_signal += info['component']

    return reconstructed_signal


def calculate_bpm_fpr(signal, fs, k1=0.3):
    """
    使用 FPR (Feature Point Recognition) 算法提取心跳率
    """
    # 限制最小间距为 4 帧（在 10Hz 下对应最高心率为 150 BPM）
    min_dist = int(fs * 0.4)
    peaks, _ = find_peaks(signal, distance=min_dist)
    troughs, _ = find_peaks(-signal, distance=min_dist)

    if len(peaks) < 2 or len(troughs) < 1:
        return 0.0

    c_max = np.max(signal[peaks])
    t_min = np.min(signal[troughs])
    delta_h_max = abs(c_max - t_min)
    th1 = k1 * delta_h_max

    # 筛选满足特征的心搏主波 (J波)
    main_waves = [p for p in peaks if (signal[p] - t_min) > th1]

    if len(main_waves) < 2:
        return 0.0

    avg_interval_frames = np.mean(np.diff(main_waves))
    bpm = (60 * fs) / avg_interval_frames
    return bpm


def calculate_bpm(signal, fs=10.0):
    """常规心跳寻峰计算方法"""
    min_dist = int(fs * 0.4)
    peaks, _ = find_peaks(signal, distance=min_dist,
                         prominence=(np.max(signal) - np.min(signal)) * 0.15)
    if len(peaks) < 2:
        return 0.0
    return (60 * fs) / np.mean(np.diff(peaks))


def smooth_heartbeat_signal(signal, window_size=7, polyorder=2):
    """
    【平滑滤波】针对心搏设计，窗口需较小（默认7帧），避免平滑掉 1-2Hz 的心跳细节
    """
    if len(signal) < window_size:
        return signal
    return savgol_filter(signal, window_size, polyorder)
