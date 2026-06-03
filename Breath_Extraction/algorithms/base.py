import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, savgol_filter, welch, butter, filtfilt
import pywt

def butter_bandpass_filter(data, lowcut=0.1, highcut=0.5, fs=10.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def wavelet_denoise(signal, alpha=0.5):
    """【预处理】自适应小波去噪：去除原始信号中的高频毛刺"""
    if len(signal) < 16: return signal
    coeffs = pywt.wavedec(signal, 'db4', level=3)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr = alpha * np.sqrt(2.0 * np.log(len(signal))) * sigma
    coeffs_dn = [coeffs[0]] + [pywt.threshold(c, thr, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs_dn, 'db4')[:len(signal)]


def calculate_snr(signal, fs=10.0, band=(0.1, 0.4)):
    """【质量评估】计算分量在呼吸频段内的信噪比 (SNR)"""
    f, psd = welch(signal, fs, nperseg=min(len(signal), 256))
    idx_band = np.logical_and(f >= band[0], f <= band[1])
    if not np.any(idx_band): return -10.0
    signal_pwr = np.sum(psd[idx_band])
    noise_pwr = np.sum(psd[~idx_band])
    return 10 * np.log10(signal_pwr / noise_pwr) if noise_pwr > 0 else 20.0

def get_dual_roi_mean(frames, window_size=5):
    """
    【自适应稳定态触发版】
    自动跳过开头的空载帧和入座时的剧烈波动，检测到稳定坐姿后才锁定中心坐标。
    """
    if len(frames) == 0: return np.array([])
    
    offset = window_size // 2
    signal_1d = []
    
    # ================= 智能寻找真正的稳定受力画面 =================
    stable_mean_frame = None
    trigger_threshold = 120   # 压力触发门限：整帧最大值超过此值，认为人已入座
    stability_window = 20    # 稳定窗口长度（约2秒）
    
    print("[ACMD-PROCESS] 开始扫描自适应坐姿稳定区间...")
    
    for i in range(len(frames) - stability_window):
        # 1. 检查当前帧是否有人坐下
        if np.max(frames[i]) > trigger_threshold:
            # 2. 验证从当前帧开始的连续 20 帧是否稳定（用标准差评估身体晃动）
            sub_series = frames[i : i + stability_window]
            # 计算这20帧的空间总均值的标准差
            frame_means = [np.mean(f) for f in sub_series]
            stability_score = np.std(frame_means)
            
            # 如果标准差很小（例如 < 5.0，说明坐下了且身体没有剧烈晃动）
            if stability_score < 5.0:
                # 抓到了真正的稳定态！计算这 20 帧的均值画面
                stable_mean_frame = np.mean(sub_series, axis=0)
                print(f"【触发成功】在第 {i} 帧捕捉到稳定坐姿（稳定度: {stability_score:.2f}）！")
                break

    # 兜底逻辑：如果全段都特别晃，找不到绝对稳定区间，就回退到用全段画面均值
    if stable_mean_frame is None:
        print("【警告】未检测到绝对稳定区间，将采用全段均值进行坐标定位。")
        stable_mean_frame = np.mean(frames, axis=0)

    # ================= 依据稳定画面锁定 ROI 坐标 =================
    l_zone = stable_mean_frame[:, :16]
    r_zone = stable_mean_frame[:, 16:]
    
    l_idx = np.unravel_index(np.argmax(l_zone), l_zone.shape)
    r_idx = np.unravel_index(np.argmax(r_zone), r_zone.shape)
    r_idx = (r_idx[0], r_idx[1] + 16)  # 修正右侧列偏移
    
    print(f"【坐标锁定】左侧臀部中心: {l_idx}, 右侧心中心: {r_idx}。开始静止降维提取...")

    # ================= 开始提取 1D 信号 =================
    for f in frames:
        def get_roi_mean(matrix, center_idx):
            r, c = center_idx
            r_s, r_e = max(0, r - offset), min(31, r + offset + 1)
            c_s, c_e = max(0, c - offset), min(31, c + offset + 1)
            roi = matrix[r_s:r_e, c_s:c_e]
            return np.mean(roi) if roi.size > 0 else 0

        signal_1d.append((get_roi_mean(f, l_idx) + get_roi_mean(f, r_idx)) / 2)

    # ================= 级联滤波链路 =================
    # 空间均值 -> 小波去噪 -> 带通滤波 (0.1-0.5Hz)
    sig_wavelet = wavelet_denoise(np.array(signal_1d), alpha=1.2)
    sig_bandpass = butter_bandpass_filter(sig_wavelet, lowcut=0.1, highcut=0.5, fs=10.0, order=3)
    
    return sig_bandpass

def reconstruct_multicomponent_with_snr(components, fs, snr_threshold=3.0):
    """
    【核心变更】全员入选逻辑：
    不再只选一个分量，而是叠加所有符合频率(0.1-0.4Hz)且 SNR 达标的分量。
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

        # 1. 频率判定：0.1 ~ 0.4 Hz
        if 0.1 <= dom_freq <= 0.4:
            # 2. 质量判定：SNR 必须大于阈值
            snr = calculate_snr(comp, fs)
            if snr >= snr_threshold:
                reconstructed_signal += comp
                found_any = True

    # 保底逻辑：如果没有任何分量达标，返回能量最大的原始分量
    return reconstructed_signal if found_any else components[0]



def calculate_bpm_fpr(signal, fs, k1=0.3):
    """
    新方法频率计算：基于文献 VMD-FPR 的 TH1 阈值识别
    """
    peaks, _ = find_peaks(signal)
    troughs, _ = find_peaks(-signal)
    if len(peaks) < 2 or len(troughs) < 1: return 0.0

    c_max = np.max(signal[peaks])
    t_min = np.min(signal[troughs])
    th1 = k1 * abs(c_max - t_min) # 计算 TH1 阈值

    # 筛选满足 TH1 条件的主波
    main_waves = [p for p in peaks if (signal[p] - t_min) > th1]
    if len(main_waves) < 2: return 0.0

    avg_interval = np.mean(np.diff(main_waves)) / fs
    return 60 / avg_interval


def calculate_bpm(signal, fs=10.0):
    """原有方法：基于 prominence 的峰值检测"""
    peaks, _ = find_peaks(signal, distance=int(fs * 1.2),
                         prominence=(np.max(signal) - np.min(signal)) * 0.2)
    if len(peaks) < 2: return 0.0
    return (60 * fs) / np.mean(np.diff(peaks))


def smooth_respiration_signal(signal, window_size=41, polyorder=3):
    if len(signal) < window_size: return signal
    return savgol_filter(signal, window_size, polyorder)


def select_best_component(components, fs):
    """筛选 0.1-0.5Hz 呼吸频段内能量最大的分量"""
    # --- 关键修复：检查 components 是否为空 ---
    if components is None or len(components) == 0:
        return np.zeros(200)  # 返回一个与 buffer 长度一致的全零数组

    best_comp = None
    max_energy = -1
    for comp in components:
        n = len(comp)
        if n == 0: continue  # 跳过空分量

        freqs = fftfreq(n, 1 / fs)[:n // 2]
        fft_vals = np.abs(fft(comp))[:n // 2]
        dom_freq = freqs[np.argmax(fft_vals)]

        if 0.1 <= dom_freq <= 0.5:
            energy = np.sqrt(np.mean(comp ** 2))
            if energy > max_energy:
                max_energy = energy
                best_comp = comp

    # --- 关键修复：确保返回值安全 ---
    if best_comp is not None:
        return best_comp
    else:
        # 如果没有找到符合频段的分量，返回第一个分量的全零版本或直接返回全零
        return np.zeros_like(components[0]) if len(components) > 0 else np.zeros(200)
    

def reconstruct_top3_components_by_energy(components, fs):
    """
    【学术策略升级】Top 3 能量自适应重构逻辑：
    筛选所有落入生理呼吸频段(0.1 ~ 0.4Hz)的分量，并按有效能量(RMS)从大到小排序，
    强力提取前 3 个核心分量进行线性叠加重构。
    """
    if components is None or len(components) == 0:
        return np.zeros(100)

    valid_components_info = []
    
    # 1. 遍历所有分量进行生理频段审查与能量测算
    for idx, comp in enumerate(components):
        n = len(comp)
        freqs = fftfreq(n, 1/fs)[:n // 2]
        fft_vals = np.abs(fft(comp))[:n // 2]
        dom_freq = freqs[np.argmax(fft_vals)]

        # 判定是否落入生理呼吸频段 (0.1 ~ 0.4 Hz)
        if 0.1 <= dom_freq <= 0.4:
            # 计算该分量的方根振幅（RMS 能量）
            rms_energy = np.sqrt(np.mean(comp ** 2))
            valid_components_info.append({
                'component': comp,
                'energy': rms_energy,
                'freq': dom_freq
            })

    # 2. 核心排序与自适应 Top 3 提取
    if len(valid_components_info) == 0:
        print("[重构警告] 未发现任何落入生理频段的分量，启动安全保底机制：取原始第一分量")
        return components[0]

    # 按能量（energy 项）从大到小（降序）进行排序
    valid_components_info.sort(key=lambda x: x['energy'], reverse=True)
    
    # 取前 3 个分量（如果满足频段的分量不足 3 个，则有几个取几个）
    top_k = min(3, len(valid_components_info))
    selected_info = valid_components_info[:top_k]
    
    print(f"[自适应重构] 成功筛选出 {len(valid_components_info)} 个生理成分，已提取 Top {top_k} 能量模态进行融合。")
    for rank, info in enumerate(selected_info):
        print(f"   -> Top {rank+1} 贡献者: 主频 = {info['freq']:.3f} Hz | RMS 能量 = {info['energy']:.6f}")

    # 3. 线性时域叠加重构
    reconstructed_signal = np.zeros_like(components[0])
    for info in selected_info:
        reconstructed_signal += info['component']

    return reconstructed_signal