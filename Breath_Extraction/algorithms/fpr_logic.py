import numpy as np


def fpr_feature_recognition(signal, fs, k1=0.3):
    """
    参考文献 VMD-FPR 逻辑实现的呼吸特征点识别
    :param signal: VMD 重构后的呼吸波形
    :param fs: 采样率
    :param k1: 主波阈值系数 (文献中用于界定主波)
    """
    # 1. 寻找所有波峰 (Ci) 和波谷 (Tj)
    from scipy.signal import find_peaks
    peaks_idx, _ = find_peaks(signal)
    troughs_idx, _ = find_peaks(-signal)

    if len(peaks_idx) == 0 or len(troughs_idx) == 0:
        return 0.0

    # 2. 计算最大峰谷差 Delta_h_max
    c_max = np.max(signal[peaks_idx])
    t_min = np.min(signal[troughs_idx])
    delta_h_max = abs(c_max - t_min)

    # 3. 定义主波阈值 TH1
    th1 = k1 * delta_h_max

    # 4. 筛选主波 (Main Wave)
    # 只有高于阈值且与最小波谷差值大于 TH1 的点才被记录
    main_waves_idx = [i for i in peaks_idx if (signal[i] - t_min) > th1]

    if len(main_waves_idx) < 2:
        return 0.0

    # 5. 计算平均周期 T_bar
    intervals = np.diff(main_waves_idx) / fs  # 转换为秒
    t_bar = np.mean(intervals)

    # 6. 计算 BPM
    bpm = 60 / t_bar
    return bpm