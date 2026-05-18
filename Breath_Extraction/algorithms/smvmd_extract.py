import numpy as np
from scipy.fft import fft, ifft, fftfreq
from .base import wavelet_denoise, calculate_snr

def SMVMD_Core(X_matrix, fs, alpha_min=1.0, alpha_max=2000.0, gamma=1.414, 
               epsilon1=1e-7, epsilon2=1e-5, max_K=4, max_iter=200):
    """
    文献《Successive multivariate variational mode decomposition...》官方 ADMM 核心算法还原
    
    参数:
    ----------
    X_matrix : ndarray
        输入的多通道时间序列，形状为 (C, T)，C 为通道数，T 为时间帧数
    fs : float
        采样率
    alpha_min : float
        正则化参数 alpha 的初始值 (文献推荐 1)
    alpha_max : float
        alpha 的最大值 (根据期望带宽计算，文献推荐 10^3 ~ 10^5)
    gamma : float
        alpha 的更新步长 (文献推荐 sqrt(2))
    epsilon1 : float
        单个模态内部迭代的收敛终止限 (文献推荐 < 10^-7)
    epsilon2 : float
        整套算法自适应停止提取的剩余能量阈值
    max_K : int
        最大允许提取的模态数 (保底门限)
        
    返回:
    ----------
    all_u : list of ndarray
        提取出的单通道 Joint IMF 列表，每个形状为 (T,)
    all_phi : list of ndarray
        对应的空间混合向量列表，每个形状为 (C, 1)
    """
    C, T = X_matrix.shape
    
    # 1. 预频域转换 (由于 VMD 在正频率域解析，取单边谱简化计算)
    # 计算全长频域谱
    X_fft = fft(X_matrix, axis=1)
    freqs = fftfreq(T, 1/fs)
    
    # 构建正频率掩膜 (DC 到奈奎斯特频率)
    half_T = T // 2
    
    # 初始化输出容器
    u_list = []
    phi_list = []
    omega_list = [] # 记录各模态的中心频率
    
    # 初始化拉格朗日乘子谱 \hat{\lambda} (形状与 X_fft 一致)
    lambda_fft = np.zeros_like(X_fft, dtype=complex)
    
    # 残差信号初始化
    X_u_fft = X_fft.copy()
    orig_energy = np.sum(np.abs(X_fft[:, :half_T])**2)
    
    # --- 外层循环：递推逐个提取模态 (Successive Extraction) ---
    for k in range(max_K):
        # 初始化当前模态的目标变量
        u_fft_curr = np.zeros(T, dtype=complex)
        phi_curr = np.random.rand(C, 1) + 1j * np.zeros((C, 1)) # 空间混合向量初始化
        phi_curr /= np.linalg.norm(phi_curr) # 归一化
        
        # 初始中心频率：取当前残差谱能量最大处 (避开 0.05Hz 以下的直流体重干扰)
        valid_idx = (freqs > 0.05) & (freqs < fs/2)
        if np.any(valid_idx):
            mean_residual_spec = np.mean(np.abs(X_u_fft[:, valid_idx]), axis=0)
            omega_curr = freqs[valid_idx][np.argmax(mean_residual_spec)]
        else:
            omega_curr = 0.2 # 默认呼吸频段中心
            
        alpha = alpha_min
        
        # --- 内层循环：ADMM 交替迭代更新变量 ---
        for it in range(max_iter):
            u_fft_old = u_fft_curr.copy()
            
            # --- 1) 更新 Joint IMF \hat{u}_k (文献公式 29) ---
            # 分子项：(phi^T) * (X + lambda/2) + alpha^2 * (w - w_k)^4 * u_old
            num_part1 = np.dot(phi_curr.conj().T, X_fft + lambda_fft / 2.0).flatten()
            num_part2 = (alpha**2) * ((freqs - omega_curr)**4) * u_fft_old
            numerator = num_part1 + num_part2
            
            # 分母项：[1 + alpha^2*(w - w_k)^4] * [1 + 2*alpha*(w - w_k)^2 + 交互惩罚项]
            denom_part1 = 1.0 + (alpha**2) * ((freqs - omega_curr)**4)
            denom_part2 = 1.0 + 2.0 * alpha * ((freqs - omega_curr)**2)
            
            # 引入历史模态谱重叠惩罚，强迫新模态与已知模态正交 (对应文献公式 29 的求和项)
            penalty_sum = np.zeros(T)
            for i, omega_past in enumerate(omega_list):
                # 防止分母为0，施加微小偏置
                penalty_sum += 1.0 / ((alpha**2) * ((freqs - omega_past)**4) + 1e-8)
            denom_part2 += penalty_sum
            
            # 更新一维联合谱 (仅在正频域计算，其余设为0)
            u_fft_curr[:half_T] = numerator[:half_T] / (denom_part1[:half_T] * denom_part2[:half_T])
            u_fft_curr[half_T:] = 0.0
            
            # --- 2) 更新空间混合向量 \phi_k (文献公式 30) ---
            phi_num = np.zeros((C, 1), dtype=complex)
            for c in range(C):
                # 积分计算：对正频域谱进行点乘叠加
                integrand = ((X_fft[c, :half_T] + lambda_fft[c, :half_T] / 2.0) * u_fft_curr[:half_T].conj()) / \
                            (1.0 + (alpha**2) * ((freqs[:half_T] - omega_curr)**4))
                phi_num[c, 0] = np.sum(integrand)
            
            # 物理特性：由于输入数据是实数压力值，空间向量取实部并实施 L2 归一化
            phi_curr = np.real(phi_num)
            phi_norm = np.linalg.norm(phi_curr)
            if phi_norm > 1e-8:
                phi_curr /= phi_norm
                
            # --- 3) 更新中心频率 \omega_k (文献公式 31) ---
            u_power = np.abs(u_fft_curr[:half_T])**2
            if np.sum(u_power) > 1e-8:
                omega_curr = np.sum(freqs[:half_T] * u_power) / np.sum(u_power)
                
            # --- 4) 双乘子梯度上升 \hat{\lambda} (文献公式 32) ---
            # 动态反馈重构误差
            for c in range(C):
                error_term = (X_fft[c, :half_T] - phi_curr[c, 0] * u_fft_curr[:half_T] + lambda_fft[c, :half_T] / 2.0) / \
                             (1.0 + (alpha**2) * ((freqs[:half_T] - omega_curr)**4))
                lambda_fft[c, :half_T] = lambda_fft[c, :half_T] + 1.0 * (error_term - lambda_fft[c, :half_T] / 2.0)
                
            # --- 5) 策略性递增带宽约束 alpha ---
            alpha = min(gamma * alpha, alpha_max)
            
            # --- 6) 内层 ADMM 收敛性检查 ---
            u_change = np.linalg.norm(u_fft_curr[:half_T] - u_fft_old[:half_T]) / (np.linalg.norm(u_fft_old[:half_T]) + 1e-8)
            if u_change < epsilon1 and alpha >= alpha_max:
                break
                
        # 结束当前模态迭代，转回时域保存
        u_time = 2.0 * np.real(ifft(u_fft_curr)) # 还原双边实信号
        
        # 保存模态参数
        u_list.append(u_time)
        phi_list.append(phi_curr)
        omega_list.append(omega_curr)
        
        # 更新未处理的残差信号谱 X_u
        for c in range(C):
            X_u_fft[c, :] -= phi_curr[c, 0] * u_fft_curr
            
        # --- 外层递推自适应终止条件 (文献公式 33 转化) ---
        residual_energy = np.sum(np.abs(X_u_fft[:, :half_T])**2)
        if (residual_energy / orig_energy) < epsilon2:
            break
            
    return u_list, phi_list

def extract_respiration(frames, fs):
    """
    对外统一的系统标准入口函数
    
    参数:
    ----------
    frames : ndarray
        经过预处理清洗后的三维压力矩阵，形状为 (N, 32, 32)
    fs : float
        采样率
    """
    N, Row, Col = frames.shape
    
    # 1. 物理重塑：将 32x32 的二维空间传感器展开为 1024 个通道的一维多元信号
    # 形状由 (N, 32, 32) -> (1024, N)
    X_multichannel = frames.reshape(N, Row * Col).T
    
    # 2. 关键工程预处理：交流耦合（去除每个通道上由于静态体重带来的巨大直流偏置）
    X_dc_removed = np.zeros_like(X_multichannel, dtype=np.float32)
    for c in range(X_multichannel.shape[0]):
        channel_sig = X_multichannel[c, :].astype(np.float32)
        if np.max(channel_sig) > 0: # 略过完全没受力的死格点
            X_dc_removed[c, :] = channel_sig - np.mean(channel_sig)
            
    # 3. 运行 SMVMD 核心迭代，剥离出公共联合模态
    # 论文实验证明，即使通道数达到 1024，由于间接更新机制，速度依然极快
    u_modes, phi_vectors = SMVMD_Core(
        X_dc_removed, fs, 
        alpha_min=1.0, 
        alpha_max=3000.0, # 根据文献公式(36)动态映射期望窄带
        max_K=4, 
        epsilon2=1e-4
    )
    
    # 4. 全员入选+质量门限重构逻辑 (完美复用你 base.py 中的优秀重构逻辑)
    reconstructed_signal = np.zeros(N)
    found_any = False
    
    for comp in u_modes:
        n = len(comp)
        freqs = fftfreq(n, 1/fs)[:n // 2]
        fft_vals = np.abs(fft(comp))[:n // 2]
        dom_freq = freqs[np.argmax(fft_vals)]
        
        # 严格限定人体生理呼吸频段: 0.1Hz ~ 0.4Hz (6 ~ 24 BPM)
        if 0.1 <= dom_freq <= 0.4:
            snr = calculate_snr(comp, fs)
            if snr >= 3.0: # SNR 达标门限
                reconstructed_signal += comp
                found_any = True
                
    # 保底机制：若无一达标，返回能量最大的第一主成分联合波形
    if not found_any and len(u_modes) > 0:
        return u_modes[0]
        
    return reconstructed_signal