"""
GOA-VMD 呼吸信号提取算法

基于论文: "Radar-based respiratory pattern recognition method using
stacking-ELCA multi-domain feature fusion" (IOP, 2025)

核心思路:
1. 使用蚱蜢优化算法 (Grasshopper Optimization Algorithm, GOA) 自适应搜索 VMD 的最优参数 (K, alpha)
2. 以平均包络熵最小化为适应度函数，促进分解出的模态分量更具周期性和稀疏性
3. 使用最优参数进行 VMD 分解
4. 通过多分量 SNR 门限重构呼吸信号

对比 vmd_MAPE.py (穷举式 MAPE 寻优):
- GOA 在连续空间搜索 alpha，而非固定 alpha=2000
- 群体智能搜索避免局部最优
- 包络熵适应度更适合周期信号分解
"""

import numpy as np
from scipy.signal import hilbert
from vmdpy import VMD

from .base import get_dual_roi_mean, reconstruct_multicomponent_with_snr, calculate_snr


# ============================================================================
# 包络熵计算
# ============================================================================

def envelope_entropy(signal):
    """
    计算信号的包络熵。

    包络熵衡量信号包络的稀疏程度/规律性:
    - 熵越低 → 包络越稀疏 → 信号越具有规则周期性 (好的分解结果)
    - 熵越高 → 包络越均匀 → 信号越接近噪声 (差的分解结果)

    Parameters
    ----------
    signal : np.ndarray, 1D
        输入信号

    Returns
    -------
    float
        包络熵值
    """
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    # 归一化为概率分布
    env_sum = np.sum(envelope)
    if env_sum < 1e-12:
        return 1e10  # 退化为零信号，返回极大惩罚
    env_norm = envelope / env_sum
    # 去除零值 (log(0) = -inf)
    env_norm = env_norm[env_norm > 1e-12]
    if len(env_norm) == 0:
        return 1e10
    entropy = -np.sum(env_norm * np.log(env_norm))
    return entropy


# ============================================================================
# GOA 适应度函数
# ============================================================================

def _goa_fitness(params, signal, fs):
    """
    GOA 适应度函数: 对给定的 VMD 参数 (K, alpha) 评估分解质量。

    适应度 = 平均包络熵 + 模态混叠惩罚 - SNR 奖励

    越低越好 (GOA 最小化目标)。

    Parameters
    ----------
    params : np.ndarray, shape (2,)
        [K (连续值，内部取整), alpha]
    signal : np.ndarray, 1D
        预处理后的 1D 呼吸信号
    fs : float
        采样频率

    Returns
    -------
    float
        适应度值 (越低越好)
    """
    # NaN 保护
    if np.any(np.isnan(signal)) or np.std(signal) < 1e-8:
        return 1e10

    K = int(np.clip(round(params[0]), 2, 10))
    alpha = int(np.clip(params[1], 500, 5000))

    try:
        u, u_hat, omega = VMD(signal, alpha=alpha, tau=0, K=K, DC=0, init=1, tol=1e-7)
    except Exception:
        return 1e10  # VMD 发散时返回极大惩罚

    # 检查分量有效性
    if any(np.any(np.isnan(comp)) for comp in u):
        return 1e10

    # 1. 平均包络熵 (核心指标)
    entropies = [envelope_entropy(comp) for comp in u]
    avg_entropy = np.mean(entropies)

    # 2. 模态混叠惩罚: 中心频率过于接近会导致模态混叠
    omega_final = omega[-1, :]  # 最终迭代的中心频率
    freq_penalty = 0.0
    if len(omega_final) > 1:
        # omega 来自 VMD 是中心角频率，转为 Hz: f = omega * fs / (2*pi)
        center_freqs_hz = np.sort(omega_final * fs / (2 * np.pi))
        freq_diffs = np.diff(center_freqs_hz)
        # 任意两个相邻中心频率差异 < 0.3Hz 则惩罚
        freq_penalty = np.sum(np.maximum(0, 0.3 - freq_diffs))

    # 3. SNR 奖励项: 呼吸频段 (0.1-0.4Hz) 内分量的 SNR 越高越好
    snr_bonus = 0.0
    for comp in u:
        snr_val = calculate_snr(comp, fs, band=(0.1, 0.4))
        if snr_val > 0:
            snr_bonus += snr_val

    # 综合适应度
    fitness = avg_entropy + 0.15 * freq_penalty - 0.005 * snr_bonus

    return fitness


# ============================================================================
# GOA (蚱蜢优化算法) 主函数
# ============================================================================

def goa_optimize(signal, fs, pop_size=12, max_iter=15, lb=None, ub=None, verbose=True,
                 progress_callback=None):
    """
    使用蚱蜢优化算法 (GOA) 搜索 VMD 最优参数。

    GOA 模拟蚱蜢群体的觅食行为:
    - 幼虫期: 移动缓慢、步幅小 (局部开发)
    - 成虫期: 长距离跳跃、快速迁移 (全局探索)

    搜索空间:
    - K: [2, 10] (整数)
    - alpha: [500, 5000] (惩罚因子)

    Parameters
    ----------
    signal : np.ndarray, 1D
        预处理后的 1D 呼吸信号
    fs : float
        采样频率
    pop_size : int
        种群规模 (默认 12)
    max_iter : int
        最大迭代次数 (默认 15)
    lb : np.ndarray or None
        搜索下界 [K_min, alpha_min]
    ub : np.ndarray or None
        搜索上界 [K_max, alpha_max]
    verbose : bool
        是否打印优化日志

    Returns
    -------
    dict
        {'K': optimal_K, 'alpha': optimal_alpha}
    """
    if lb is None:
        lb = np.array([2.0, 500.0])
    if ub is None:
        ub = np.array([10.0, 5000.0])

    dim = 2  # 搜索维度: K, alpha
    n = pop_size

    # --- 初始化种群 ---
    np.random.seed(42)
    population = np.zeros((n, dim))
    population[:, 0] = np.random.randint(int(lb[0]), int(ub[0]) + 1, n)  # K 离散
    population[:, 1] = np.random.uniform(lb[1], ub[1], n)                 # alpha 连续

    # --- 评估初始适应度 ---
    fitness = np.array([_goa_fitness(population[i], signal, fs) for i in range(n)])

    best_idx = np.argmin(fitness)
    best_position = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    if verbose:
        print(f"[GOA] 初始最优: K={int(best_position[0])}, alpha={int(best_position[1])}, "
              f"适应度={best_fitness:.4f}")

    # GOA 参数
    cmax = 1.0      # c_max: 控制探索-开发平衡
    cmin = 0.00004  # c_min

    for iteration in range(max_iter):
        # 报告进度 (用于 UI 进度条)
        if progress_callback:
            progress_callback(iteration + 1, max_iter)

        # 更新舒适区缩减系数 c (线性递减)
        c = cmax - iteration * (cmax - cmin) / max_iter

        new_population = np.zeros_like(population)

        for i in range(n):
            # 计算社会交互力 S_i
            S_i = np.zeros(dim)
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(population[i] - population[j])
                    if dist < 1e-8:
                        dist = 1e-8
                    # 社会力函数 S(r) = f * exp(-r/l) - exp(-r)
                    # f: 吸引强度, l: 吸引尺度
                    f_attr = 0.5
                    l_attr = 1.5
                    s_r = f_attr * np.exp(-dist / l_attr) - np.exp(-dist)
                    # 方向向量
                    direction = (population[j] - population[i]) / dist
                    S_i += s_r * direction

            # 位置更新: X_i^{new} = c * S_i + T_d   (T_d = 当前全局最优)
            new_population[i] = c * S_i + best_position

        # 边界约束
        new_population[:, 0] = np.clip(np.round(new_population[:, 0]), lb[0], ub[0])
        new_population[:, 1] = np.clip(new_population[:, 1], lb[1], ub[1])

        # 与上一代混合：保留上一代最优个体 (精英保留)
        new_population[0] = best_position.copy()

        population = new_population

        # 评估新一代适应度
        fitness = np.array([_goa_fitness(population[i], signal, fs) for i in range(n)])

        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_fitness:
            best_fitness = fitness[min_idx]
            best_position = population[min_idx].copy()
            if verbose:
                print(f"[GOA] 第{iteration+1:2d}代: K={int(best_position[0])}, "
                      f"alpha={int(best_position[1])}, 适应度={best_fitness:.4f}")

    K_opt = int(np.clip(round(best_position[0]), 2, 10))
    alpha_opt = int(np.clip(best_position[1], 500, 5000))

    if verbose:
        print(f"[GOA] 优化完成: K={K_opt}, alpha={alpha_opt}, 最佳适应度={best_fitness:.4f}")

    return {'K': K_opt, 'alpha': alpha_opt}


# ============================================================================
# 主入口函数
# ============================================================================

def extract_respiration(frames, fs, progress_callback=None):
    """
    GOA-VMD 呼吸信号提取。

    1. 空间降维 + 小波去噪 + 带通滤波 (由 get_dual_roi_mean 完成)
    2. GOA 自适应优化 VMD 参数 (K, alpha)
    3. 使用最优参数进行 VMD 分解
    4. 多分量 SNR 门限重构呼吸波形

    Parameters
    ----------
    frames : np.ndarray, shape (N, 32, 32)
        压力矩阵序列
    fs : float
        采样频率 (Hz)
    progress_callback : callable or None
        进度回调, 签名为 callback(current_iter, max_iter)

    Returns
    -------
    np.ndarray
        重构后的 1D 呼吸信号
    """
    # 1. 空间降维 + 预处理 (小波去噪 + 0.1-0.5Hz 带通滤波)
    signal_1d = get_dual_roi_mean(frames)

    if len(signal_1d) < 100:
        return signal_1d

    # NaN 保护: 如果 get_dual_roi_mean 返回 NaN (小波去噪对极低幅值信号可能产生 NaN)，
    # 回退到简单空间均值 + 带通滤波
    if np.any(np.isnan(signal_1d)):
        print("[GOA-VMD] 检测到 NaN, 回退到简单空间均值预处理")
        from .base import butter_bandpass_filter
        # 简单全局空间均值
        signal_1d = np.mean(frames, axis=(1, 2))
        # 轻量级去噪
        signal_1d = butter_bandpass_filter(signal_1d, lowcut=0.1, highcut=0.5, fs=fs, order=3)
        if np.any(np.isnan(signal_1d)) or np.std(signal_1d) < 1e-8:
            return signal_1d

    # 2. GOA 自适应优化 VMD 参数
    best_params = goa_optimize(signal_1d, fs, pop_size=8, max_iter=10, verbose=True,
                                progress_callback=progress_callback)
    K_opt = best_params['K']
    alpha_opt = best_params['alpha']

    # 3. VMD 分解
    u, u_hat, omega = VMD(signal_1d, alpha=alpha_opt, tau=0, K=K_opt, DC=0, init=1, tol=1e-7)

    # 4. 多分量 SNR 门限重构
    result = reconstruct_multicomponent_with_snr(u, fs)

    return result
