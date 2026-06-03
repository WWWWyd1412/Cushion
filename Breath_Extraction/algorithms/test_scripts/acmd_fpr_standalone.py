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
from scipy.signal import welch

# 修正导入链路
import data_loader
import preprocess
from algorithms import base

# 模拟或导入 ACMD 核心算法
def ACMD_Core(signal, fs, max_components=7, tol=1e-4):
    """
    自适应啁啾模型分解 (ACMD) 核心实现
    无需预设固定的 K 值，根据残差能量自适应剥离分量
    """
    components = []
    residual = signal.copy()
    orig_energy = np.sum(signal ** 2)
    
    for i in range(max_components):
        # 1. 初始化当前分量的瞬时频率 (以当前残差的傅里叶主频作为初始估计)
        n = len(residual)
        fft_vals = np.abs(np.fft.fft(residual))[:n // 2]
        freqs = np.fft.fftfreq(n, 1/fs)[:n // 2]
        
        # 忽略傅里叶变换前几个代表直流和极低频基线漂移的点
        valid_idx = freqs > 0.05  
        if np.any(valid_idx):
            filtered_fft = fft_vals[valid_idx]
            filtered_freqs = freqs[valid_idx]
            init_freq = filtered_freqs[np.argmax(filtered_fft)]
        else:
            init_freq = 0.2  # 默认呼吸中心频率

        # 2. 时频脊线变换与解调解算
        t = np.arange(n) / fs
        # 建立正余弦解调基底
        c = np.cos(2 * np.pi * init_freq * t)
        s = np.sin(2 * np.pi * init_freq * t)
        
        # 最小二乘自适应滤波拟合主能量分量 IA
        comp_i = c * (np.dot(residual, c) / (np.dot(c, c) + 1e-6)) + s * (np.dot(residual, s) / (np.dot(s, s) + 1e-6))
        
        # 3. 剥离已提取能量
        residual -= comp_i
        components.append(comp_i)
        
        # 4. 自适应终止检查
        current_mape = np.sum(residual ** 2) / (orig_energy + 1e-12)
        if current_mape < tol:
            break
            
    return np.array(components), residual


class AcmdFprStepByStep(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ACMD-FPR 逻辑验证工具 - 全模态 PSD 增强版")
        self.resize(1600, 1000)

        # 核心数据成员
        self.fs = 10.0
        self.clean_frames = None
        self.signal_1d = None
        self.components_cache = None
        self.residual_cache = None
        
        # 存储两种不同的重构结果用于步骤 5 对比测速
        self.results = {
            "single_best": {"recon": None, "bpm": 0.0},
            "top3_energy": {"recon": None, "bpm": 0.0}
        }

        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # 控制面板
        control_panel = QVBoxLayout()
        self.btn_1 = QPushButton("步骤 1: 数据加载与清洗")
        self.btn_2 = QPushButton("步骤 2: 左右分区 5x5 ROI 提取")
        self.btn_3 = QPushButton("步骤 3: ACMD 自适应能量流剥离与主频计算")
        self.btn_4 = QPushButton("步骤 4: 生理频段 Top 3 能量重构与波形对比")
        self.btn_5 = QPushButton("步骤 5: FPR 呼吸节律判定与双策略对比")
        # === 新增步骤 6 按钮 ===
        self.btn_6 = QPushButton("步骤 6: ACMD 模态功率谱密度 (PSD) 分析")

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
        self.log_edit.append(f"<b>[ACMD-INFO]</b> {msg}")
        QApplication.processEvents()

    def run_step_1(self):
        """步骤 1: 加载与清洗"""
        path, _ = QFileDialog.getOpenFileName(self, "选择数据", "", "Text Files (*.txt)")
        if not path: return
        try:
            t, f = data_loader.load_pressure_txt(path)
            _, self.clean_frames = preprocess.clean_dataset(t, f)
            self.log(f"数据加载成功: {len(self.clean_frames)} 帧")
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.imshow(self.clean_frames[len(self.clean_frames) // 2], cmap='jet')
            ax.set_title("预处理后的中间帧热力图")
            self.canvas.draw()
        except Exception as e:
            self.log(f"加载出错: {str(e)}")

    def run_step_2(self):
        """步骤 2: 左右分区 ROI 提取"""
        if self.clean_frames is None: return
        raw_signal = base.get_dual_roi_mean(self.clean_frames, window_size=5)
        
        # 减去均值，强制将静态体重带来的直流分量归零
        self.signal_1d = raw_signal - np.mean(raw_signal)
        
        self.log("5x5 ROI 信号提取完成 (已含小波去噪，并完成交流耦合去直流)")
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.signal_1d, color='#0078d7')
        ax.set_title("1D ROI 均值趋势信号")
        self.canvas.draw()

    def run_step_3(self):
        """步骤 3: 能量剥离与主频同步解算"""
        if self.signal_1d is None: return
        self.log("执行 ACMD 变分贪婪解调计算...")

        sig = self.signal_1d
        self.components_cache, self.residual_cache = ACMD_Core(sig, self.fs, max_components=7, tol=0.0001)
        
        num_modes = len(self.components_cache)
        self.log(f"ACMD 自动收敛终止。成功剥离出 {num_modes} 个自适应啁啾模态(Modes)")

        self.figure.clear()
        k_range = list(range(1, num_modes + 1))
        energies = []

        for idx, comp in enumerate(self.components_cache):
            rms_energy = np.sqrt(np.mean(comp ** 2))
            energies.append(rms_energy)
            
            n = len(comp)
            fft_vals = np.abs(np.fft.fft(comp))[:n // 2]
            freqs = np.fft.fftfreq(n, 1 / self.fs)[:n // 2]
            
            dom_freq_hz = freqs[np.argmax(fft_vals)]
            dom_freq_bpm = dom_freq_hz * 60.0  
            
            self.log(f"Mode {idx+1} | 有效均方根能量: {rms_energy:.6f} | 脊线主频: {dom_freq_hz:.3f} Hz ({dom_freq_bpm:.1f} BPM)")

        ax_energy = self.figure.add_subplot(111)
        ax_energy.bar(k_range, energies, color='#2c3e50', alpha=0.8, width=0.5)
        ax_energy.set_title("ACMD 各级自适应啁啾模态有效能量分布")
        ax_energy.set_xlabel("Mode 序号")
        ax_energy.set_ylabel("RMS 振幅能量")
        ax_energy.grid(True, alpha=0.3)
        self.canvas.draw()

    def run_step_4(self):
        """步骤 4: ACMD 生理带内 Top 3 能量选择重构"""
        if self.components_cache is None: return
        self.log("开始对比重构解算：策略 A (传统单主频) vs 策略 B (新：生理带内能量 Top 3 融合)...")

        self.results["single_best"]["recon"] = base.select_best_component(self.components_cache, self.fs)
        self.results["top3_energy"]["recon"] = base.reconstruct_top3_components_by_energy(self.components_cache, self.fs)

        offset = 100
        for key in ["single_best", "top3_energy"]:
            recon_sig = self.results[key]["recon"]
            if len(recon_sig) > offset:
                processed = recon_sig[offset:]
            else:
                processed = recon_sig
            self.results[key]["recon"] = base.smooth_respiration_signal(processed)

        self.figure.clear()
        ax1 = self.figure.add_subplot(211)
        ax1.plot(self.results["single_best"]["recon"], color='#d35400', linewidth=2)
        ax1.set_title("策略 A: ACMD 经典单主频最大能量分量重构波形")
        ax1.grid(True, alpha=0.3)

        ax2 = self.figure.add_subplot(212)
        ax2.plot(self.results["top3_energy"]["recon"], color='#27ae60', linewidth=2)
        ax2.set_title("策略 B (新): 生理带内自适应能量 Top 3 分量重构波形")
        ax2.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
        self.log("双策略重构与滤波平滑完成。")

    def run_step_5(self):
        """步骤 5: FPR 特征峰值比识别与频率计算"""
        if self.results["single_best"]["recon"] is None: return
        self.log("正在将两种重构波形同时注入 FPR 特征识别算法...")

        bpm_s = base.calculate_bpm_fpr(self.results["single_best"]["recon"], self.fs)
        bpm_t3 = base.calculate_bpm_fpr(self.results["top3_energy"]["recon"], self.fs)

        self.results["single_best"]["bpm"] = bpm_s
        self.results["top3_energy"]["bpm"] = bpm_t3

        self.figure.axes[0].set_title(f"策略 A: 单主频极大分量 | 提取节律: {bpm_s:.2f} BPM")
        self.figure.axes[1].set_title(f"策略 B (新): 能量 Top 3 融合重构 | 提取节律: {bpm_t3:.2f} BPM")
        self.canvas.draw()
        self.log(f"【判定完毕】: 单主频成分法检测结果 = {bpm_s:.2f} BPM; 能量 Top 3 融合策略检测结果 = {bpm_t3:.2f} BPM")

    def run_step_6(self):
        """=== 步骤 6: ACMD 功率谱密度 (PSD) 分析与主峰识别 ==="""
        if self.components_cache is None:
            self.log("错误: 请先执行步骤 3 分解出模态信号！")
            return
        
        self.log("正在使用 Welch 方法计算 ACMD 各个解调模态的功率谱密度 (PSD)...")
        self.figure.clear()
        
        num_modes = len(self.components_cache)
        # 动态调配子图颜色池
        colors = ['#2980b9', '#e67e22', '#27ae60', '#9b59b6', '#34495e', '#16a085', '#d35400']
        
        for idx, comp in enumerate(self.components_cache):
            # 1. 计算加窗分段平均功率谱密度
            nperseg = min(len(comp), 256)
            f, psd = welch(comp, self.fs, nperseg=nperseg)
            
            # 2. 限定生理核心关注带 (0.05 Hz ~ 1.0 Hz) 阻击低频姿势漂移和高频干扰
            valid_mask = (f >= 0.05) & (f <= 1.0)
            f_roi = f[valid_mask]
            psd_roi = psd[valid_mask]
            
            # 3. 寻找生理带内的最大谱线波峰值
            if len(psd_roi) > 0:
                peak_idx = np.argmax(psd_roi)
                peak_freq = f_roi[peak_idx]
                peak_val = psd_roi[peak_idx]
                peak_bpm = peak_freq * 60.0
                self.log(f"Mode {idx+1} -> 生理带最强谱峰: 频率 = {peak_freq:.3f} Hz ({peak_bpm:.1f} BPM) | 能量峰值 = {peak_val:.4e}")
            else:
                peak_freq, peak_val, peak_bpm = 0, 0, 0
            
            # 4. 并行创建多层子图
            ax = self.figure.add_subplot(num_modes, 1, idx + 1)
            ax.plot(f, psd, color=colors[idx % len(colors)], linewidth=2, label=f"Mode {idx+1} PSD")
            
            # 如果主波峰落在标准生理呼吸带内 (0.1 ~ 0.4 Hz)，打星并画线高亮标记
            if 0.1 <= peak_freq <= 0.4:
                ax.scatter(peak_freq, peak_val, color='red', s=40, zorder=5)
                ax.axvline(x=peak_freq, color='red', linestyle='--', alpha=0.5)
                ax.text(peak_freq + 0.02, peak_val * 0.7, f"呼吸主峰: {peak_bpm:.1f} BPM\n(谱值: {peak_val:.2e})", 
                        color='red', fontweight='bold', fontsize=9)
            
            ax.set_xlim(0, 1.5) # 展示到 1.5 Hz，涵盖完整的呼吸谐波和早期跳动段
            ax.set_ylabel("PSD")
            ax.set_title(f"Mode {idx+1} 功率谱密度响应 (RMS能量: {np.sqrt(np.mean(comp**2)):.4f})", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
        self.figure.axes[-1].set_xlabel("Frequency (Hz)")
        self.figure.tight_layout()
        self.canvas.draw()
        self.log("ACMD 全模态功率谱密度图谱绘制完成。")


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    app = QApplication(sys.argv)
    window = AcmdFprStepByStep()
    window.show()
    sys.exit(app.exec_())