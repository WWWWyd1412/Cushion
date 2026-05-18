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
                             QWidget, QPushButton, QTextEdit, QFileDialog, QLabel)
from PyQt5.QtCore import Qt

# 导入底层链路
import data_loader
import preprocess
from algorithms import base, smvmd_extract


class SmvmdFprStepByStep(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SMVMD-FPR 逻辑验证工具 - 多元自适应递推版")
        self.resize(1600, 1000)

        # 核心数据成员
        self.fs = 10.0
        self.clean_frames = None
        self.X_multichannel = None
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
        self.btn_2 = QPushButton("步骤 2: 1024通道展开与交流耦合(去体重)")
        self.btn_3 = QPushButton("步骤 3: 递推自适应分解 (无需预设 K)")
        self.btn_4 = QPushButton("步骤 4: 多组分 SNR 门限重构")
        self.btn_5 = QPushButton("步骤 5: FPR 呼吸节律判定")

        for btn in [self.btn_1, self.btn_2, self.btn_3, self.btn_4, self.btn_5]:
            btn.setFixedHeight(50)
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

    def log(self, msg):
        self.log_edit.append(f"<b>[SMVMD-INFO]</b> {msg}")
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
        """步骤 2: 通道变换与交流耦合去直流"""
        if self.clean_frames is None: return
        N, Row, Col = self.clean_frames.shape
        
        # 降维重塑：(N, 32, 32) -> (1024, N)
        X_raw = self.clean_frames.reshape(N, Row * Col).T
        
        # 减去时间域均值，去除静态体重的直流偏置 (这是防止低频漂移干扰核心中心频率更新的关键)
        self.X_multichannel = np.zeros_like(X_raw, dtype=np.float32)
        active_channels = 0
        for c in range(X_raw.shape[0]):
            channel_sig = X_raw[c, :].astype(np.float32)
            if np.max(channel_sig) > 0:
                self.X_multichannel[c, :] = channel_sig - np.mean(channel_sig)
                active_channels += 1
                
        self.log(f"多通道物理展开完成。当前受力活跃通道数: {active_channels}/1024")
        
        # 可视化展示前 5 个活跃通道的时域波形
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        plot_count = 0
        for c in range(1024):
            if np.std(self.X_multichannel[c, :]) > 5.0:
                ax.plot(self.X_multichannel[c, :], label=f"Ch {c}")
                plot_count += 1
                if plot_count >= 4: break
        ax.set_title("去除直流后的部分活跃通道信号波形 (交流耦合)")
        ax.legend()
        self.canvas.draw()

    def run_step_3(self):
        """步骤 3: 运行 SMVMD 官方迭代核心"""
        if self.X_multichannel is None: return
        self.log("开始执行多元递推变分求解（ADMM正弦/余弦基底交替投影）...")
        
        # 调用 smvmd_extract 中的官方复原核心算法 (限制最大搜索 4 个模态，能量阈值 1e-4)
        self.u_modes, self.phi_vectors = smvmd_extract.SMVMD_Core(
            self.X_multichannel, self.fs, 
            alpha_min=1.0, alpha_max=3000.0, 
            epsilon2=1e-4, max_K=4
        )
        
        num_extracted = len(self.u_modes)
        self.log(f"SMVMD 自适应收敛终止！成功剥离出 {num_extracted} 个联合模态。")
        
        # 绘制所有剥离出的单通道 Joint IMFs 进行对比
        self.figure.clear()
        for i in range(min(num_extracted, 4)):
            ax = self.figure.add_subplot(num_extracted, 1, i + 1)
            ax.plot(self.u_modes[i], color='#7f8c8d' if i>0 else '#2980b9')
            ax.set_title(f"Joint IMF {i+1} 时域波形")
            ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()
        
        for idx, u in enumerate(self.u_modes):
            # 评估其能量有效性 (RMS)
            rms = np.sqrt(np.mean(u ** 2))
            self.log(f"模态 {idx+1} | 联合方根振幅: {rms:.4f}")

    def run_step_4(self):
        """步骤 4: 生理频段叠加重构"""
        if self.u_modes is None: return
        self.log("注入生理学门限验证：筛选满足 0.1~0.4Hz 且 SNR >= 3dB 的分量...")
        
        N = self.X_multichannel.shape[1]
        self.reconstructed_breath = np.zeros(N)
        found_any = False
        
        for idx, comp in enumerate(self.u_modes):
            n = len(comp)
            freqs = np.fft.fftfreq(n, 1/self.fs)[:n // 2]
            fft_vals = np.abs(np.fft.fft(comp))[:n // 2]
            dom_freq = freqs[np.argmax(fft_vals)]
            
            # 生理限制
            if 0.1 <= dom_freq <= 0.4:
                snr = base.calculate_snr(comp, self.fs)
                self.log(f"-> 模态 {idx+1}: 主频 = {dom_freq:.3f} Hz, 信噪比 SNR = {snr:.2f} dB")
                if snr >= 3.0:
                    self.reconstructed_breath += comp
                    found_any = True
                    self.log(f"   [状态]: 通过考核，累加进重构呼吸信号")
                else:
                    self.log(f"   [状态]: 信噪比未达标，剔除")
            else:
                self.log(f"-> 模态 {idx+1}: 主频 = {dom_freq:.3f} Hz [状态]: 超出呼吸生理频段，剔除")
                
        if not found_any:
            self.log("警告: 未找到任何完全达标的分量，启动保底机制：取第一级主成分联合模态")
            self.reconstructed_breath = self.u_modes[0]
            
        # 切除起始前 100 帧过渡区并调用 Savitzky-Golay 平滑
        offset = 100
        if len(self.reconstructed_breath) > offset:
            processed = self.reconstructed_breath[offset:]
        else:
            processed = self.reconstructed_breath
        self.reconstructed_breath = base.smooth_respiration_signal(processed)
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.reconstructed_breath, color='#16a085', linewidth=2)
        ax.set_title("SMVMD 多维空间权重学习 - 重构后的标准一维呼吸波形")
        ax.grid(True, alpha=0.3)
        self.canvas.draw()
        self.log("最终一维呼吸波形自适应平滑及起始噪声切除完成。")

    def run_step_5(self):
        """步骤 5: 基于文献特征点阈值的频率识别"""
        if self.reconstructed_breath is None: return
        self.log("启动 FPR (Feature Point Recognition) 算法进行周期解算...")
        
        # 依靠你的优秀公式对处理后的最纯净一维信号进行 BPM 测算
        bpm = base.calculate_bpm_fpr(self.reconstructed_breath, self.fs)
        
        self.figure.axes[0].set_title(f"SMVMD 重构呼吸波形 | 精准监测频率: {bpm:.2f} BPM")
        self.canvas.draw()
        self.log(f"【核心判定结果】: 当前受试者的生理呼吸率 = {bpm:.2f} 次/分 (BPM)")


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    app = QApplication(sys.argv)
    window = SmvmdFprStepByStep()
    window.show()
    sys.exit(app.exec_())