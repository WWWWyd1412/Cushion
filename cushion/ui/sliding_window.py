"""
滑窗分析引擎
============
提供滑窗分析的核心逻辑，供所有 GUI 入口共用。

功能:
    - 按指定窗口/步长生成分析窗口
    - 重叠相加 (Overlap-Add) 波形融合
    - Tukey 窗 + 相位对齐自动拼接
    - BPM 趋势统计
"""

import numpy as np
from scipy.signal import find_peaks


def generate_windows(total_frames, window_size, step_size):
    """
    生成滑窗索引列表。

    Parameters
    ----------
    total_frames : int
        总帧数。
    window_size : int
        每个窗口的帧数。
    step_size : int
        窗口滑动步长（帧数）。

    Returns
    -------
    list of (start_idx, end_idx)
        每个窗口的起止帧索引。
    """
    windows = []
    for start in range(0, total_frames - window_size + 1, step_size):
        windows.append((start, start + window_size))
    return windows


def overlap_add_fusion(waveforms, windows, total_frames, fs=10.0):
    """
    重叠相加波形融合 — 使用 Tukey 窗 + 相位对齐。

    Parameters
    ----------
    waveforms : list of ndarray
        每个窗口的 1D 波形 (长度可能不同)。
    windows : list of (start, end)
        每个窗口对应的帧索引。
    total_frames : int
        总帧数。
    fs : float
        采样率 (用于相位对齐的互相关计算)。

    Returns
    -------
    ndarray, 1D — 融合后的全长波形。
    """
    fused = np.zeros(total_frames)
    weight = np.zeros(total_frames)

    for (start, end), wave in zip(windows, waveforms):
        seg_len = end - start
        if len(wave) < seg_len:
            # 对齐到窗口长度
            padded = np.zeros(seg_len)
            padded[:len(wave)] = wave
            wave = padded
        elif len(wave) > seg_len:
            wave = wave[:seg_len]

        # Tukey 窗 (alpha=0.25) 用于平滑重叠过渡
        tukey = _tukey_window(seg_len, alpha=0.25)

        # 相位对齐: 与前一个重叠段进行互相关对齐
        overlap_start = max(0, start)
        if weight[overlap_start:start].max() > 0 and start > 0:
            shift = _find_phase_shift(
                fused[max(0, start - seg_len // 2):start + seg_len // 2],
                wave,
                fs
            )
            if shift != 0:
                wave = np.roll(wave, shift)

        fused[start:end] += wave * tukey
        weight[start:end] += tukey

    # 归一化
    mask = weight > 1e-8
    fused[mask] /= weight[mask]
    return fused


def _tukey_window(length, alpha=0.25):
    """生成 Tukey (余弦渐缩) 窗。"""
    from scipy.signal import tukey as _tukey
    return _tukey(length, alpha=alpha)


def _find_phase_shift(segment_a, segment_b, fs):
    """
    通过互相关找到 segment_b 相对于 segment_a 的最佳相位偏移。

    Returns
    -------
    int — 偏移帧数。
    """
    if len(segment_a) < 3 or len(segment_b) < 3:
        return 0
    min_len = min(len(segment_a), len(segment_b))
    a = segment_a[:min_len]
    b = segment_b[:min_len]
    a = (a - np.mean(a)) / (np.std(a) + 1e-8)
    b = (b - np.mean(b)) / (np.std(b) + 1e-8)

    cross_corr = np.correlate(a, b, mode='full')
    max_lag = min(20, len(a) // 4)
    mid = len(cross_corr) // 2
    search_region = cross_corr[mid - max_lag:mid + max_lag + 1]
    shift = np.argmax(search_region) - max_lag
    return shift


def compute_bpm_statistics(bpm_values):
    """
    计算 BPM 序列的统计信息。

    Parameters
    ----------
    bpm_values : list of float

    Returns
    -------
    dict — {'mean', 'std', 'min', 'max', 'count', 'valid_ratio'}
    """
    valid = [b for b in bpm_values if b > 0]
    if not valid:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0,
                'count': 0, 'valid_ratio': 0.0}

    return {
        'mean': np.mean(valid),
        'std': np.std(valid),
        'min': np.min(valid),
        'max': np.max(valid),
        'count': len(valid),
        'valid_ratio': len(valid) / len(bpm_values),
    }
