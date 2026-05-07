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
        self.setWindowTitle("VMD-FPR 算法全流程分步可视化 (时间轴版)")
        self.resize(1600, 900)
        
        self.fs = 10.0 # 采样频率 10Hz
        self.raw_frames = None
        self.clean_frames = None
        self.signal_1d = None
        self.results_dict = {}
        self.best_k_rebound = 2
        self.best_k_fast = 2
        
        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        control_panel = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        
        self.btn_load = QPushButton("步骤 1: 加载与清洗数据")
        self.btn_roi = QPushButton("步骤 2: 动态 5x5 ROI 提取")
        self.btn_vmd = QPushButton("步骤 3: VMD 分解与能量分析")
        self.btn_reconstruct = QPushButton("步骤 4: 呼吸波形重构与平滑")
        self.btn_fpr = QPushButton("步骤 5: FPR 特征识别与 BPM 计算")

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

        self.figure = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        main_layout.addLayout(plot_layout, 3)

        self.btn_load.clicked.connect(self.step1_load)
        self.btn_roi.clicked.connect(self.step2_roi)
        self.btn_vmd.clicked.connect(self.step3_vmd)
        self.btn_reconstruct.clicked.connect(self.step4_reconstruct)
        self.btn_fpr.clicked.connect(self.step5_fpr)

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
            # 转换为时间轴
            time_axis = np.arange(len(self.clean_frames)) / self.fs
            ax.plot(time_axis, np.mean(self.clean_frames, axis=(1, 2)), color='gray')
            ax.set_title("1. 空间全局平均趋势 (时间轴)")
            ax.set_xlabel("时间 (s)")
            self.canvas.draw()
            self.btn_roi.setEnabled(True)

    def step2_roi(self):
        self.signal_1d = get_dual_roi_mean(self.clean_frames, window_size=5)
        self.log("动态 5x5 ROI 提取完成。")
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        time_axis = np.arange(len(self.signal_1d)) / self.fs
        ax.plot(time_axis, self.signal_1d, color='blue')
        ax.set_title("2. 动态 ROI 提取后的 1D 信号")
        ax.set_xlabel("时间 (s)")
        self.canvas.draw()
        self.btn_vmd.setEnabled(True)

    def step3_vmd(self):
        from vmdpy import VMD
        self.log("<b>开始 VMD 分解与能量饱和分析...</b>")
        
        mapes = []
        k_range = list(range(2, 11))
        all_results = {}

        for k in k_range:
            u, _, _ = VMD(self.signal_1d, alpha=2000, tau=0, K=k, DC=0, init=1, tol=1e-7)
            res = self.signal_1d - np.sum(u, axis=0)
            mape = np.sum(res ** 2) / np.sum(self.signal_1d ** 2)
            mapes.append(mape)
            all_results[k] = (u, mape)
            self.log(f"K={k} | MAPE: {mape:.6f}")

        # 逻辑判断
        self.best_k_rebound = 2
        for i in range(1, len(mapes)):
            if mapes[i] > mapes[i-1]:
                self.best_k_rebound = k_range[i-1]
                break
            self.best_k_rebound = k_range[i]
        
        diffs = np.abs(np.diff(mapes))
        self.best_k_fast = k_range[np.argmax(diffs) + 1]
        self.results_dict = all_results

        self.plot_energy_analysis(mapes, k_range, all_results[self.best_k_rebound][0])
        self.btn_reconstruct.setEnabled(True)

    def plot_energy_analysis(self, mapes, k_range, u_best):
        self.figure.clear()
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax1.plot(k_range, mapes, 'o-', color='#2c3e50')
        ax1.axvline(x=self.best_k_rebound, color='red', linestyle='--')
        ax1.set_title("VMD 分解能量残差比 (MAPE)")
        ax1.set_xlabel("分解层数 K")
        
        ax2 = self.figure.add_subplot(2, 1, 2)
        energies = [np.sum(imf**2) for imf in u_best]
        ratios = [e / np.sum(energies) * 100 for e in energies]
        ax2.bar([f"IMF{i+1}" for i in range(len(ratios))], ratios, color='#3498db')
        ax2.set_title(f"K={self.best_k_rebound} 时各模态能量占比 (%)")
        self.figure.tight_layout()
        self.canvas.draw()

    def step4_reconstruct(self):
        from algorithms.vmd_MAPE import reconstruct_respiration_signal
        
        def get_final_sig(u):
            recon = reconstruct_respiration_signal(u, self.fs)
            sig = recon[100:] if len(recon) > 100 else recon # 切除前100帧
            return smooth_respiration_signal(sig, window_size=25, polyorder=3)

        u_reb, _ = self.results_dict[self.best_k_rebound]
        u_fst, _ = self.results_dict[self.best_k_fast]
        
        self.sig_rebound = get_final_sig(u_reb)
        self.sig_fast = get_final_sig(u_fst)

        self.figure.clear()
        # 处理时间轴（注意重构后切掉了前100帧）
        time_axis_reb = np.arange(len(self.sig_rebound)) / self.fs
        time_axis_fst = np.arange(len(self.sig_fast)) / self.fs

        ax1 = self.figure.add_subplot(2, 1, 1)
        ax1.plot(time_axis_reb, self.sig_rebound, color='#e74c3c')
        ax1.set_title(f"方法 A: 反弹判定重构 (K={self.best_k_rebound})")
        ax1.set_xlabel("时间 (s)")
        
        ax2 = self.figure.add_subplot(2, 1, 2)
        ax2.plot(time_axis_fst, self.sig_fast, color='#3498db')
        ax2.set_title(f"方法 B: 快速下降重构 (K={self.best_k_fast})")
        ax2.set_xlabel("时间 (s)")
        
        self.figure.tight_layout()
        self.canvas.draw()
        self.btn_fpr.setEnabled(True)

    def step5_fpr(self):
        from scipy.signal import find_peaks
        
        def process_fpr(signal):
            p, _ = find_peaks(signal)
            t, _ = find_peaks(-signal)
            if len(p) == 0 or len(t) == 0: return 0.0, []
            th1 = 0.3 * abs(np.max(signal[p]) - np.min(signal[t]))
            main_waves = [idx for idx in p if (signal[idx] - np.min(signal[t])) > th1]
            if len(main_waves) < 2: return 0.0, main_waves
            bpm = (60 * self.fs) / np.mean(np.diff(main_waves))
            return bpm, main_waves

        bpm_a, peaks_a = process_fpr(self.sig_rebound)
        bpm_b, peaks_b = process_fpr(self.sig_fast)

        self.figure.clear()
        # 绘图逻辑同样应用时间轴
        for i, (sig, peaks, bpm, k, col) in enumerate([
            (self.sig_rebound, peaks_a, bpm_a, self.best_k_rebound, '#e74c3c'),
            (self.sig_fast, peaks_b, bpm_b, self.best_k_fast, '#3498db')
        ]):
            ax = self.figure.add_subplot(2, 1, i+1)
            time_axis = np.arange(len(sig)) / self.fs
            ax.plot(time_axis, sig, color=col)
            ax.plot(np.array(peaks)/self.fs, sig[peaks], "kx") # 特征点也要除以 fs
            ax.set_title(f"结果 {chr(65+i)}: {bpm:.1f} BPM (K={k})")
            ax.set_xlabel("时间 (s)")

        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    app = QApplication(sys.argv)
    window = VmdFprStepByStep()
    window.show()
    sys.exit(app.exec_())