import numpy as np
from vmdpy import VMD
from scipy.fft import fft, fftfreq
from .base import get_dual_roi_mean


def optimize_vmd_with_mape(signal, fs=10.0):
    """
    结合文献 VMD-FPR 逻辑：基于 MAPE 自动寻找最优 K 值并重构呼吸信号
    """
    if len(signal) < 100:
        return signal

    mapes = []
    k_range = range(2, 11)  # 文献实验中 K 的范围扩展到 2-10
    best_u = None
    best_k = 2

    for k in k_range:
        # VMD 分解
        u, u_hat, omega = VMD(signal, alpha=2000, tau=0, K=k, DC=0, init=1, tol=1e-7)

        # 计算残差能量比 MAPE (式 7)
        res = signal - np.sum(u, axis=0)
        mape = np.sum(res ** 2) / np.sum(signal ** 2)
        mapes.append(mape)

        # 拐点检测逻辑：如果 MAPE 出现回升，说明出现过分解，停止迭代[cite: 10]
        if len(mapes) > 1:
            if mapes[-1] > mapes[-2]:
                break

        best_u = u
        best_k = k

    # 修改点：调用重构函数，而不是只选一个分量
    return reconstruct_respiration_signal(best_u, fs)


def reconstruct_respiration_signal(components, fs):
    """
    文献 VMD-FPR 步骤 2：选择所有有效 IMF 分量并重构信号[cite: 10]
    不再是只选能量最大的 IMF，而是将所有处于呼吸频段 (0.1-0.5Hz) 的分量叠加
    """
    reconstructed_signal = np.zeros_like(components[0])
    found_valid_comp = False

    for comp in components:
        n = len(comp)
        # 计算主频
        freqs = fftfreq(n, 1 / fs)[:n // 2]
        fft_vals = np.abs(fft(comp))[:n // 2]
        dom_freq = freqs[np.argmax(fft_vals)]

        # 呼吸频段判定：0.1 ~ 0.5 Hz[cite: 10]
        # 如果该分量落入呼吸频段，则将其累加进结果信号中
        if 0.1 <= dom_freq <= 0.4:
            reconstructed_signal += comp
            found_valid_comp = True

    # 如果没找到合适的呼吸分量，返回全阵列的最后一个 IMF (通常是最低频部分) 作为保底
    if not found_valid_comp:
        return components[-1]

    return reconstructed_signal


def calculate_bpm_fpr(signal, fs, k1=0.3):
    """
    文献 3.3 节 FPR (Feature Point Recognition) 核心算法实现[cite: 10]
    用于替代原本简单的 find_peaks 逻辑
    """
    from scipy.signal import find_peaks

    # 1. 寻找所有波峰 (Ci) 和波谷 (Tj)[cite: 10]
    peaks, _ = find_peaks(signal)
    troughs, _ = find_peaks(-signal)

    if len(peaks) < 2 or len(troughs) < 1:
        return 0.0

    # 2. 计算最大峰谷差 Delta_h_max (式 11)[cite: 10]
    c_max = np.max(signal[peaks])
    t_min = np.min(signal[troughs])
    delta_h_max = abs(c_max - t_min)

    # 3. 计算主波阈值 TH1 (k1 一般取 0.3-0.5)[cite: 10]
    th1 = k1 * delta_h_max

    # 4. 识别主波特征点：峰值与最小谷底之差必须大于 TH1[cite: 10]
    main_waves = [p for p in peaks if (signal[p] - t_min) > th1]

    if len(main_waves) < 2:
        return 0.0

    # 5. 计算平均周期 T_bar 并转换为 BPM[cite: 10]
    avg_interval_frames = np.mean(np.diff(main_waves))
    bpm = (60 * fs) / avg_interval_frames

    return bpm


def extract_respiration(frames, fs):
    """
    新方法入口：使用动态选点 + VMD_MAPE 重构[cite: 11, 14]
    """
    signal_1d = get_dual_roi_mean(frames)
    return optimize_vmd_with_mape(signal_1d, fs)