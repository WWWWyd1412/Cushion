import numpy as np
from scipy.fft import fft, ifft
from .base import get_multi_roi_signals, fuse_signals_ica, reconstruct_multicomponent_with_snr


def mvmd(X, alpha=2000, tau=0, K=4, DC=0, init=1, tol=1e-7, max_iter=200):
    """
    多变量变分模态分解 (Multivariate Variational Mode Decomposition, MVMD)
    
    Parameters
    ----------
    X : np.ndarray, shape (M, T)
        多通道输入信号，M 为通道数 (ROI 数量)，T 为时间采样点数
    alpha : float
        带宽约束参数 (平衡参数)
    tau : float
        对偶更新步长 (0 表示无重建约束的经典模式)
    K : int
        模态分解个数
    DC : int
        是否保留第一分量为 DC 部分 (0: 否, 1: 是)
    init : int
        中心频率初始化策略 (1: 均匀分布, 2: 对数分布, 其他: 随机分布)
    tol : float
        收敛精度阈值
    max_iter : int
        最大迭代次数

    Returns
    -------
    u_time : np.ndarray, shape (K, M, T)
        分解出的时域模态分量，第 k 个模态对应所有通道的振幅分量
    omega : np.ndarray, shape (N_iter, K)
        迭代收敛的中心频率历史
    """
    M, T = X.shape
    
    # 1. 信号镜像延拓 (Mirroring) 避免边缘截断效应
    half_T = T // 2
    X_padded = np.zeros((M, 2 * T))
    for m in range(M):
        X_padded[m, :half_T] = X[m, :half_T][::-1]
        X_padded[m, half_T:half_T + T] = X[m, :]
        X_padded[m, half_T + T:] = X[m, T - half_T:][::-1]
        
    T_padded = X_padded.shape[1]
    
    # 2. 频域坐标
    freqs = np.arange(T_padded) / T_padded
    
    # 3. 快速傅里叶变换到频域
    X_fft = fft(X_padded, axis=1)
    
    # 仅使用正频部分 (解析信号单边谱进行计算，最后重构时乘以 2 并取实部)
    half_len = T_padded // 2
    X_fft_half = X_fft[:, :half_len]
    freqs_half = freqs[:half_len]
    
    # 4. 初始化中心频率 omega
    omega = np.zeros((max_iter, K))
    if init == 1:
        # 均匀初始化
        for k in range(K):
            omega[0, k] = 0.5 * k / K
    elif init == 2:
        # 对数初始化
        omega[0, :] = np.logspace(np.log10(0.01), np.log10(0.5), K)
    else:
        # 随机初始化
        omega[0, :] = np.sort(np.random.rand(K) * 0.5)
        
    if DC == 1:
        omega[0, 0] = 0.0
        
    # 5. 初始化频域模态与对偶乘子
    u_fft = np.zeros((K, M, half_len), dtype=complex)
    lambda_fft = np.zeros((M, half_len), dtype=complex)
    
    # 6. ADMM 交替方向乘子法迭代优化
    it = 0
    converged = False
    u_diff = np.zeros(max_iter)
    
    while it < max_iter - 1 and not converged:
        u_fft_old = u_fft.copy()
        
        # 6.1 更新每个模态的频域值 u_k
        for k in range(K):
            # 去除当前模态后，其他所有模态之和
            sum_other = np.sum(u_fft, axis=0) - u_fft[k, :, :]
            
            # 当前模态中心频率带宽权重分母
            denom = 1.0 + 2.0 * alpha * (freqs_half - omega[it, k])**2
            
            for m in range(M):
                numerator = X_fft_half[m, :] - sum_other[m, :] - lambda_fft[m, :] / 2.0
                u_fft[k, m, :] = numerator / denom
                
            # 6.2 更新跨通道共享的中心频率 omega_k
            if not (DC == 1 and k == 0):
                # 融合所有通道当前模态的能量分布
                power_spectrum_sum = np.sum(np.abs(u_fft[k, :, :])**2, axis=0)
                denom_freq = np.sum(power_spectrum_sum)
                if denom_freq > 1e-12:
                    omega[it + 1, k] = np.sum(freqs_half * power_spectrum_sum) / denom_freq
                else:
                    omega[it + 1, k] = omega[it, k]
            else:
                omega[it + 1, k] = 0.0
                
        # 6.3 更新对偶乘子 lambda
        lambda_fft += tau * (X_fft_half - np.sum(u_fft, axis=0))
        
        # 6.4 收敛性检查
        diff_sum = 0.0
        norm_sum = 0.0
        for k in range(K):
            diff_sum += np.sum(np.abs(u_fft[k] - u_fft_old[k])**2)
            norm_sum += np.sum(np.abs(u_fft_old[k])**2)
            
        u_diff[it] = diff_sum / (norm_sum + 1e-12)
        if u_diff[it] < tol:
            converged = True
            
        it += 1
        
    # 7. 频域逆变换重构时域信号
    u_fft_full = np.zeros((K, M, T_padded), dtype=complex)
    u_fft_full[:, :, :half_len] = u_fft
    
    u_time = np.zeros((K, M, T))
    for k in range(K):
        for m in range(M):
            # 重构解析信号并反裁切回原始信号大小
            analytic = ifft(u_fft_full[k, m, :])
            u_time[k, m, :] = 2.0 * np.real(analytic[half_T : half_T + T])
            
    return u_time, omega[:it, :]


def extract_respiration(frames, fs, K=4, alpha=2000):
    """
    MVMD 多通道呼吸信号提取入口
    """
    # 1. 提取多 ROI 信号 (默认提取 4 个核心象限受力区域)
    multi_signals = get_multi_roi_signals(frames, num_rois=4, window_size=5)
    if len(multi_signals) == 0:
        return np.zeros(len(frames))
        
    # 2. 运行 MVMD 算法进行时空联合模态分解
    # multi_signals shape: (4, T)
    # u_time shape: (K, 4, T)
    u_time, _ = mvmd(multi_signals, alpha=alpha, tau=0, K=K, DC=0, init=1, tol=1e-7)
    
    # 3. 对每一级模态进行多通道 FastICA 时空盲源分离融合
    fused_components = []
    for k in range(K):
        # 对第 k 个模态的所有通道 u_time[k] (shape: 4 x T) 进行 FastICA 提取
        fused_k = fuse_signals_ica(u_time[k], fs)
        fused_components.append(fused_k)
        
    # 4. 生理频率门限与 SNR 多分量重构
    return reconstruct_multicomponent_with_snr(np.array(fused_components), fs)
