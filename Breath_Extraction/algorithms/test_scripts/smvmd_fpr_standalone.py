import sys
import os
import numpy as np
import matplotlib

# --- 动态将父目录加入 sys.path 确保跨目录导入正常 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 强制使用更稳定的 Agg 后端进行绘图计算
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QTextEdit, QFileDialog, QLabel, QProgressDialog)
from PyQt5.QtCore import Qt
from scipy.signal import welch

# 导入底层链路
import data_loader
import preprocess
from algorithms import base, smvmd_extract


class SmvmdFprStepByStep(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SMVMD-FPR 逻辑验证工具 - 功率谱 PSD 增强版")
        self.resize(1600, 1000)

        # 核心数据成员
        self.fs = 10.0
        self.clean_frames = None
        self.X_shrunk = None
        self.u_modes = None
        self.phi_vectors = None
        self.reconstructed_breath = None

        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # 控制面板
        control_panel = QVBoxLayout()
        self.btn_1 = QPushButton("步骤 1: 数据加载与清洗")
        self.btn_2 = QPushButton("步骤 2: 空间脱水与交流耦合 (去体重/剔死区)")
        self.btn_3 = QPushButton("步骤 3: 向量化递推分解 (带实时进度条)")
        self.btn_4 = QPushButton("步骤 4: 生理频段 SNR 重构")
        self.btn_5 = QPushButton("步骤 5: FPR 呼吸节律判定")
        # === 新增步骤 6 按钮 ===
        self.btn_6 = QPushButton("步骤 6: 模态功率谱密度 (PSD) 分析")

        for btn in [self.btn_1, self.btn_2, self.btn_3, self.btn_4, self.btn_5, self.btn_6]:
            btn.setFixedHeight(45)
            control_panel.addWidget(btn)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setStyleSheet("background-color: #2b2b2b; color: #a9b7c6; font-family: 'Consolas';")
        control_panel.addWidget(QLabel("执行日志:"))
        control_panel.addWidget(self.log_edit)
        layout.addLayout(control_panel, 1)

        # 绘图区域
        self.figure = plt.figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 3)

        self.btn_1.clicked.connect(self.run_step_1)
        self.btn_2.clicked.connect(self.run_step_2)
        self.btn_3.clicked.connect(self.run_step_3)
        self.btn_4.clicked.connect(self.run_step_4)
        self.btn_5.clicked.connect(self.run_step_5)
        self.btn_6.clicked.connect(self.run_step_6) # 绑定槽函数

    def log(self, msg):
        self.log_edit.append(f"<b>[SMVMD-O1]</b> {msg}")
        QApplication.processEvents()

    def run_step_1(self):
        """步骤 1: 加载与清洗"""
        path, _ = QFileDialog.getOpenFileName(self, "选择数据", "", "Text Files (*.txt)")
        if not path: return
        try:
            t, f = data_loader.load_pressure_txt(path)
            _, self.clean_frames = preprocess.clean_dataset(t, f)
            self.log(f"数据清洗完成，当前有效帧数: {len(self.clean_frames)} 帧")
            
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.imshow(self.clean_frames[len(self.clean_frames) // 2], cmap='jet')
            ax.set_title("中间帧压力分布热力图")
            self.canvas.draw()
        except Exception as e:
            self.log(f"加载出错: {str(e)}")

    def run_step_2(self):
        """步骤 2: 空间物理解耦方案"""
        if self.clean_frames is None: return
        N, Row, Col = self.clean_frames.shape
        X_raw = self.clean_frames.reshape(N, Row * Col).T
        
        X_active_list = []
        for c in range(X_raw.shape[0]):
            channel_sig = X_raw[c, :].astype(np.float32)
            if np.std(channel_sig) > 13:
                X_active_list.append(channel_sig - np.mean(channel_sig))
                
        self.X_shrunk = np.array(X_active_list)
        self.log(f"【空间脱水完成】: 剔除了无信号格点，通道总数由 1024 精简为 {self.X_shrunk.shape[0]}")
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        for i in range(min(4, self.X_shrunk.shape[0])):
            ax.plot(self.X_shrunk[i, :], label=f"Active_Ch {i+1}")
        ax.set_title("去直流及死区后的多通道交流耦合波形")
        ax.legend()
        self.canvas.draw()

    def run_step_3(self):
        """步骤 3: 运行带实时进度条刷新的 SMVMD"""
        if self.X_shrunk is None: return
        self.log("配置解调参数，正在初始化交互式进度条算子...")
        
        progress_dialog = QProgressDialog("正在初始化多元解调...", "取消解算", 0, 100, self)
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle("SMVMD 多维时空并行矩阵计算")
        progress_dialog.setMinimumDuration(0)
        progress_dialog.resize(500, 120)
        progress_dialog.setStyleSheet("""
            QProgressDialog { background-color: #f3f3f3; font-size: 14px; }
            QProgressBar { border: 1px solid #bbb; border-radius: 5px; text-align: center; height: 22px; font-weight: bold; }
            QProgressBar::chunk { background-color: #2980b9; width: 8px; }
            QPushButton { padding: 5px 15px; border-radius: 4px; border: 1px solid #c8c8c8; background: #e1e1e1; }
            QPushButton:hover { background: #d1d1d1; }
        """)
        progress_dialog.show()
        QApplication.processEvents()

        def on_algo_progress(current_step, total_steps, current_k, current_it):
            if progress_dialog.wasCanceled():
                return False
            percentage = int((current_step / total_steps) * 100)
            progress_dialog.setValue(percentage)
            progress_dialog.setLabelText(f"正在逐层剥离核心呼吸成分... [ 模态 K = {current_k} | 迭代数 = {current_it} ]")
            QApplication.processEvents()
            return True

        self.log("启动多元变分级联算法...")
        
        self.u_modes, self.phi_vectors = smvmd_extract.SMVMD_Core(
            self.X_shrunk, self.fs, 
            alpha_min=1.0, alpha_max=3000.0, 
            epsilon2=1e-4, max_K=4, max_iter=50,
            progress_callback=on_algo_progress
        )
        
        progress_dialog.close()
        num_extracted = len(self.u_modes)
        self.log(f"分解执行完毕！算法自动收敛提取出 {num_extracted} 个联合模态。")
        
        self.figure.clear()
        for i in range(num_extracted):
            ax = self.figure.add_subplot(num_extracted, 1, i + 1)
            ax.plot(self.u_modes[i], color='#2980b9' if i==0 else '#e67e22')
            ax.set_title(f"Joint IMF {i+1} 时域特征波形")
            ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

    def run_step_4(self):
        """步骤 4: 质量门限重构"""
        if self.u_modes is None: return
        self.log("执行生理约束审查与 SNR 达标度过滤...")
        
        N = self.X_shrunk.shape[1]
        self.reconstructed_breath = np.zeros(N)
        found_any = False
        
        for idx, comp in enumerate(self.u_modes):
            n = len(comp)
            freqs = np.fft.fftfreq(n, 1/self.fs)[:n // 2]
            fft_vals = np.abs(np.fft.fft(comp))[:n // 2]
            dom_freq = freqs[np.argmax(fft_vals)]
            
            if 0.1 <= dom_freq <= 0.4:
                snr = base.calculate_snr(comp, self.fs)
                self.log(f"-> 模态 {idx+1}: 主频 = {dom_freq:.3f} Hz, 信噪比 = {snr:.2f} dB")
                if snr >= 3.0:
                    self.reconstructed_breath += comp
                    found_any = True
                    self.log("   [状态]: 准予重构")
                else:
                    self.log("   [状态]: 噪声过高，剔除")
            else:
                self.log(f"-> 模态 {idx+1}: 主频 = {dom_freq:.3f} Hz [状态]: 超出生理频段，丢弃")
                
        if not found_any:
            self.log("未找到完美达标项，应用保底策略：提取主分量模态")
            self.reconstructed_breath = self.u_modes[0]
            
        offset = 100
        if len(self.reconstructed_breath) > offset:
            processed = self.reconstructed_breath[offset:]
        else:
            processed = self.reconstructed_breath
        self.reconstructed_breath = base.smooth_respiration_signal(processed)
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.reconstructed_breath, color='#2c3e50', linewidth=2)
        ax.set_title("SMVMD 自适应空间融合重构后的一维纯净呼吸信号")
        ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def run_step_5(self):
        """步骤 5: 频率测算"""
        if self.reconstructed_breath is None: return
        self.log("注入 FPR 周期门限提取主波周期...")
        bpm = base.calculate_bpm_fpr(self.reconstructed_breath, self.fs)
        self.figure.axes[0].set_title(f"SMVMD 空间自适应权重融合提取波形 | 提取呼吸率: {bpm:.2f} BPM")
        self.canvas.draw()
        self.log(f"【判定完毕】: 受试者当前生理呼吸率测量结果为: {bpm:.2f} 次/分 (BPM)")

    def run_step_6(self):
        """=== 新增步骤 6: 模态功率谱密度 (PSD) 分析与主峰识别 ==="""
        if self.u_modes is None:
            self.log("错误: 请先执行步骤 3 分解出模态信号！")
            return
        
        self.log("正在使用 Welch 方法计算各个解调模态的功率谱密度 (PSD)...")
        self.figure.clear()
        
        num_modes = len(self.u_modes)
        colors = ['#2980b9', '#e67e22', '#27ae60', '#9b59b6']
        
        for idx, comp in enumerate(self.u_modes):
            # 1. 计算功率谱密度，设置合理的窗长（如256点）以确保频率分辨率
            nperseg = min(len(comp), 256)
            f, psd = welch(comp, self.fs, nperseg=nperseg)
            
            # 2. 划定生理呼吸核心关注带 (0.05 Hz ~ 1.0 Hz)，防止极低偏置和极高噪点带偏主峰识别
            valid_mask = (f >= 0.05) & (f <= 1.0)
            f_roi = f[valid_mask]
            psd_roi = psd[valid_mask]
            
            # 3. 寻找能量最高的核心主波峰值及物理频率
            if len(psd_roi) > 0:
                peak_idx = np.argmax(psd_roi)
                peak_freq = f_roi[peak_idx]
                peak_val = psd_roi[peak_idx]
                peak_bpm = peak_freq * 60
                self.log(f"Joint IMF {idx+1} -> 生理带最强谱峰: 频率 = {peak_freq:.3f} Hz ({peak_bpm:.1f} BPM) | 能量峰值 = {peak_val:.4e}")
            else:
                peak_freq, peak_val, peak_bpm = 0, 0, 0
            
            # 4. 并行绘制子图
            ax = self.figure.add_subplot(num_extracted := num_modes, 1, idx + 1)
            ax.plot(f, psd, color=colors[idx % len(colors)], linewidth=2, label=f"IMF {idx+1} PSD")
            
            # 如果主波峰落在经典生理呼吸区域内（0.1 ~ 0.4 Hz），进行高亮标定
            if 0.1 <= peak_freq <= 0.4:
                ax.scatter(peak_freq, peak_val, color='red', s=40, zorder=5)
                ax.axvline(x=peak_freq, color='red', linestyle='--', alpha=0.5)
                ax.text(peak_freq + 0.02, peak_val * 0.8, f"呼吸主峰: {peak_bpm:.1f} BPM\n(Val: {peak_val:.2e})", 
                        color='red', fontweight='bold', fontsize=9)
            
            ax.set_xlim(0, self.fs/2) # 展示到奈奎斯特截止频率
            ax.set_ylabel("Power Spectrum Density")
            ax.set_title(f"Joint IMF {idx+1} 功率谱响应 (最高峰值: {peak_val:.3e})")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
        self.figure.axes[-1].set_xlabel("Frequency (Hz)")
        self.figure.tight_layout()
        self.canvas.draw()
        self.log("全模态功率谱密度图谱绘制完成，已成功完成频域主峰标记。")


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    app = QApplication(sys.argv)
    window = SmvmdFprStepByStep()
    window.show()
    sys.exit(app.exec_())