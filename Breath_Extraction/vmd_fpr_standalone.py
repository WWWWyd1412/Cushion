import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QTextEdit)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# 核心模块导入
import data_loader
import preprocess
from algorithms import vmd_MAPE
from algorithms.base import get_dual_roi_mean, calculate_bpm_fpr, smooth_respiration_signal

class VmdFprStepByStep(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VMD-FPR 算法全流程分步可视化 (科研专用)")
        self.resize(1600, 900)
        
        self.fs = 10.0
        self.raw_frames = None
        self.clean_frames = None
        self.signal_1d = None
        self.reconstructed = None
        self.final_signal = None
        
        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- 左侧控制台 ---
        control_panel = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("算法执行日志...")
        
        # 流程按钮
        self.btn_load = QPushButton("步骤 1: 加载与清洗数据")
        self.btn_roi = QPushButton("步骤 2: 动态 5x5 ROI 提取")
        self.btn_vmd = QPushButton("步骤 3: VMD 分解与 MAPE 优化")
        self.btn_reconstruct = QPushButton("步骤 4: 呼吸波形重构与平滑")
        self.btn_fpr = QPushButton("步骤 5: FPR 特征识别与 BPM 计算")

        # 初始禁用后续步骤
        for btn in [self.btn_roi, self.btn_vmd, self.btn_reconstruct, self.btn_fpr]:
            btn.setEnabled(False)

        control_panel.addWidget(QLabel("<b>算法执行步骤:</b>"))
        control_panel.addWidget(self.btn_load)
        control_panel.addWidget(self.btn_roi)
        control_panel.addWidget(self.btn_vmd)
        control_panel.addWidget(self.btn_reconstruct)
        control_panel.addWidget(self.btn_fpr)
        control_panel.addWidget(QLabel("<b>过程日志:</b>"))
        control_panel.addWidget(self.log_output)
        main_layout.addLayout(control_panel, 1)

        # --- 右侧绘图区 (分多个子图) ---
        self.figure = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        main_layout.addLayout(plot_layout, 3)

        # 信号连接
        self.btn_load.clicked.connect(self.step1_load)
        self.btn_roi.clicked.connect(self.step2_roi)
        self.btn_vmd.clicked.connect(self.step3_vmd)
        self.btn_reconstruct.clicked.connect(self.step4_reconstruct)
        self.btn_fpr.clicked.connect(self.step5_fpr)

    # --- 步骤实现 ---

    def log(self, text):
        self.log_output.append(f"<b>[LOG]</b>: {text}")

    def step1_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择压力数据", "", "Text Files (*.txt)")
        if path:
            t, f = data_loader.load_pressure_txt(path)
            _, self.clean_frames = preprocess.clean_dataset(t, f)
            self.log(f"数据加载完成。有效帧数: {len(self.clean_frames)}")
            
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(np.mean(self.clean_frames, axis=(1, 2)), color='gray')
            ax.set_title("1. 空间全局平均趋势 (含大量噪声)")
            self.canvas.draw()
            self.btn_roi.setEnabled(True)

    def step2_roi(self):
        self.signal_1d = get_dual_roi_mean(self.clean_frames, window_size=5)
        self.log("动态 5x5 ROI 提取完成。已追踪左右臀部受力中心。")
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.signal_1d, color='blue')
        ax.set_title("2. 动态 ROI 提取后的 1D 信号 (去趋势)")
        self.canvas.draw()
        self.btn_vmd.setEnabled(True)

    def step3_vmd(self):
        from vmdpy import VMD
        self.log("<b>开始对比 VMD 优化策略...</b>")
        
        mapes = []
        k_range = range(2, 11)
        all_results = {}

        # --- 阶段 1: 数据采集 ---
        for k in k_range:
            u, _, _ = VMD(self.signal_1d, alpha=2000, tau=0, K=k, DC=0, init=1, tol=1e-7)
            res = self.signal_1d - np.sum(u, axis=0)
            mape = np.sum(res ** 2) / np.sum(self.signal_1d ** 2)
            mapes.append(mape)
            all_results[k] = (u, mape)
            self.log(f"K={k} | MAPE: {mape:.4f}")

        # --- 阶段 2: 逻辑判定 ---
        # 逻辑 A: 反弹判定
        best_k_rebound = 2
        for i in range(1, len(mapes)):
            if mapes[i] > mapes[i-1]:
                best_k_rebound = k_range[i-1]
                break
            best_k_rebound = k_range[i]
        self.log(f"<span style='color:red;'>[反弹判定] 锁定 K={best_k_rebound}</span>")

        # 逻辑 B: 快速下降判定
        diffs = np.abs(np.diff(mapes))
        max_diff_idx = np.argmax(diffs)
        best_k_fast = k_range[max_diff_idx + 1] 
        self.log(f"<span style='color:blue;'>[快速下降判定] 锁定 K={best_k_fast}</span>")

        # --- 核心修正：保存变量供下一步使用 ---
        self.results_dict = all_results  # 解决 AttributeError
        self.best_k_rebound = best_k_rebound
        self.best_k_fast = best_k_fast

        # --- 阶段 3: 可视化对比 ---
        self.show_vmd_dual_comparison(all_results, best_k_rebound, best_k_fast)
        self.btn_reconstruct.setEnabled(True)

    def show_vmd_dual_comparison(self, results, k_rebound, k_fast):
        """
        在同一个画布上对比两种逻辑选出的 IMF 分量[cite: 10, 19]
        """
        self.figure.clear()
        
        # 左侧展示反弹逻辑的结果
        u_reb, m_reb = results[k_rebound]
        for i in range(k_rebound):
            ax = self.figure.add_subplot(max(k_rebound, k_fast), 2, 2*i + 1)
            ax.plot(u_reb[i], color='green', linewidth=0.7)
            ax.set_ylabel(f"R-IMF{i+1}", fontsize=7)
            if i == 0: ax.set_title(f"反弹逻辑 (K={k_rebound})")

        # 右侧展示快速下降逻辑的结果
        u_fst, m_fst = results[k_fast]
        for i in range(k_fast):
            ax = self.figure.add_subplot(max(k_rebound, k_fast), 2, 2*i + 2)
            ax.plot(u_fst[i], color='blue', linewidth=0.7)
            ax.set_ylabel(f"F-IMF{i+1}", fontsize=7)
            if i == 0: ax.set_title(f"快速下降逻辑 (K={k_fast})")
            
        self.canvas.draw()


    def step4_reconstruct(self):
        """
        对比两种 K 值选取逻辑下的重构结果
        """
        self.log("<b>开始对比重构逻辑...</b>")
        
        # 假设你在 Step 3 已经保存了 best_k_rebound 和 best_k_fast
        # 这里我们模拟调用 vmd_MAPE 的重构逻辑[cite: 11, 21]
        
        def get_reconstructed_sig(u, fs):
            """执行重构与强平滑的内部函数[cite: 11, 14, 21]"""
            from algorithms.vmd_MAPE import reconstruct_respiration_signal
            recon = reconstruct_respiration_signal(u, fs)
            
            # 切除前 100 帧干扰[cite: 19]
            offset = 100
            sig_clipped = recon[offset:] if len(recon) > offset else recon
            
            # 应用 25 窗口的 SG 滤波解决毛刺问题[cite: 14]
            return smooth_respiration_signal(sig_clipped, window_size=25, polyorder=3)

        # 1. 获取两种逻辑下的重构信号[cite: 11, 21]
        # self.results_dict 存储了 Step 3 计算的各层 IMF
        u_reb, _ = self.results_dict[self.best_k_rebound]
        u_fst, _ = self.results_dict[self.best_k_fast]
        
        sig_rebound = get_reconstructed_sig(u_reb, self.fs)
        sig_fast = get_reconstructed_sig(u_fst, self.fs)
        
        # 保存供下一步使用（默认以论文推荐的反弹法为准）
        self.final_signal = sig_rebound 

        # 2. 绘图对比[cite: 19]
        self.figure.clear()
        
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax1.plot(sig_rebound, color='#e74c3c', label=f"反弹判定 (K={self.best_k_rebound})")
        ax1.set_title(f"方法 A: 反弹判定重构波形 (更倾向于提取微弱特征)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = self.figure.add_subplot(2, 1, 2)
        ax2.plot(sig_fast, color='#3498db', label=f"快速下降 (K={self.best_k_fast})")
        ax2.set_title(f"方法 B: 快速下降重构波形 (更倾向于锁定主导波动)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        self.canvas.draw()
        self.log(f"重构完成。对比发现：K={self.best_k_rebound} 包含的分量更多，波形形态可能更完整 。")
        self.btn_fpr.setEnabled(True)

    def step5_fpr(self):
        """
        步骤 5: 分别对比两种 K 值选取模式下的 FPR 识别结果
        """
        self.log("<b>开始执行双模式 FPR 特征识别...</b>")
        from scipy.signal import find_peaks

        def get_bpm_and_peaks(signal, fs, k1=0.3):
            """内部工具：识别特征点并计算 BPM"""
            peaks_idx, _ = find_peaks(signal)
            troughs_idx, _ = find_peaks(-signal)
            
            if len(peaks_idx) == 0 or len(troughs_idx) == 0:
                return 0.0, []
            
            # 计算动态阈值 TH1
            c_max = np.max(signal[peaks_idx])
            t_min = np.min(signal[troughs_idx])
            th1 = k1 * abs(c_max - t_min)
            
            # 筛选符合 TH1 条件的主波峰[cite: 11]
            main_waves = [p for p in peaks_idx if (signal[p] - t_min) > th1]
            
            if len(main_waves) < 2:
                return 0.0, main_waves
                
            # 计算平均周期 T_bar[cite: 11, 16]
            intervals = np.diff(main_waves) / fs
            bpm = 60 / np.mean(intervals)
            return bpm, main_waves

        # 1. 重新获取两种逻辑下的平滑信号（为了确保绘图数据完整）[cite: 11]
        # 注意：这里需要我们在 step4 中保存 sig_rebound 和 sig_fast
        # 如果你没保存，可以在这里快速重算
        u_reb, _ = self.results_dict[self.best_k_rebound]
        u_fst, _ = self.results_dict[self.best_k_fast]
        
        # 假设重构和平滑逻辑已在 step4 验证通过[cite: 14, 21]
        from algorithms.vmd_MAPE import reconstruct_respiration_signal
        
        # 反弹法重构与平滑
        recon_reb = reconstruct_respiration_signal(u_reb, self.fs)
        sig_reb = smooth_respiration_signal(recon_reb[100:], 25, 3)
        
        # 快速下降法重构与平滑
        recon_fst = reconstruct_respiration_signal(u_fst, self.fs)
        sig_fst = smooth_respiration_signal(recon_fst[100:], 25, 3)

        # 2. 计算 BPM 和特征点[cite: 11, 16]
        bpm_reb, peaks_reb = get_bpm_and_peaks(sig_reb, self.fs)
        bpm_fst, peaks_fst = get_bpm_and_peaks(sig_fst, self.fs)

        # 3. 绘图对比[cite: 19]
        self.figure.clear()
        
        # 上图：反弹法结果
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax1.plot(sig_reb, color='#e74c3c', label='波形 (Rebound)')
        ax1.plot(peaks_reb, sig_reb[peaks_reb], "x", color='black', markersize=8, label='FPR 特征点')
        ax1.set_title(f"方法 A (反弹 K={self.best_k_rebound}): {bpm_reb:.1f} BPM")
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 下图：快速下降法结果
        ax2 = self.figure.add_subplot(2, 1, 2)
        ax2.plot(sig_fst, color='#3498db', label='波形 (Fast-Drop)')
        ax2.plot(peaks_fst, sig_fst[peaks_fst], "x", color='black', markersize=8, label='FPR 特征点')
        ax2.set_title(f"方法 B (快速下降 K={self.best_k_fast}): {bpm_fst:.1f} BPM")
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        self.canvas.draw()

        # 4. 日志总结[cite: 10, 11]
        self.log(f"<b>[结果对比]</b>")
        self.log(f"反弹法 (K={self.best_k_rebound}) 呼吸率: <span style='color:red;'>{bpm_reb:.1f} BPM</span>")
        self.log(f"快速下降 (K={self.best_k_fast}) 呼吸率: <span style='color:blue;'>{bpm_fst:.1f} BPM</span>")
        
        if abs(bpm_reb - bpm_fst) < 0.5:
            self.log("结论：两种方法判定结果一致，说明当前信号质量较好，呼吸特征非常稳定。")
        else:
            self.log("结论：结果存在差异。通常反弹判定法由于包含更多细节，在低信噪比下更为准确。快速下降法可能过于简化，适合信号质量较高的情况。")

if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    app = QApplication(sys.argv)
    window = VmdFprStepByStep()
    window.show()
    sys.exit(app.exec_())