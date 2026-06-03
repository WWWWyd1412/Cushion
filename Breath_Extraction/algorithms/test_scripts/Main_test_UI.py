import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 强制使用 Qt5 绘图后端
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QMessageBox, QComboBox, QGroupBox, QLineEdit,
                             QCheckBox, QTextEdit)
from PyQt5.QtCore import Qt

# --- 动态添加父目录到 sys.path 以支持跨包导入 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 导入底层核心数据处理与算法包
import data_loader
import preprocess
from algorithms import base, extract_emd, extract_vmd


# ================= 核心 ACMD 算法定义 =================
def ACMD_Core(signal, fs, max_components=5, tol=1e-4):
    """
    自适应啁啾模型分解 (ACMD) 核心迭代解调算法
    """
    components = []
    residual = signal.copy()
    orig_energy = np.sum(signal ** 2)
    
    for i in range(max_components):
        n = len(residual)
        fft_vals = np.abs(np.fft.fft(residual))[:n // 2]
        freqs = np.fft.fftfreq(n, 1/fs)[:n // 2]
        
        # 忽略极低频直流与基线漂移
        valid_idx = freqs > 0.05  
        if np.any(valid_idx):
            filtered_fft = fft_vals[valid_idx]
            filtered_freqs = freqs[valid_idx]
            init_freq = filtered_freqs[np.argmax(filtered_fft)]
        else:
            init_freq = 0.2  # 默认呼吸频率 0.2Hz

        # 时频脊线解调自适应滤波
        t = np.arange(n) / fs
        c = np.cos(2 * np.pi * init_freq * t)
        s = np.sin(2 * np.pi * init_freq * t)
        
        # 最小二乘求解幅值并重建当前分量
        comp_i = c * (np.dot(residual, c) / (np.dot(c, c) + 1e-6)) + s * (np.dot(residual, s) / (np.dot(s, s) + 1e-6))
        
        residual -= comp_i
        components.append(comp_i)
        
        # 能量收敛终止条件
        current_mape = np.sum(residual ** 2) / (orig_energy + 1e-12)
        if current_mape < tol:
            break
            
    return np.array(components), residual


def extract_acmd(frames, fs):
    """
    ACMD 呼吸提取接口：定位 ROI -> 解调分解 -> Top 3 生理频段重构
    """
    # 1. 通过稳定态定位算法提取 1D 信号并完成去噪和滤波
    signal_1d = base.get_dual_roi_mean(frames, window_size=5)
    # 2. 交流耦合，减去直流基线
    signal_1d = signal_1d - np.mean(signal_1d)
    
    # 3. 运行 ACMD 剥离出主要模态
    components, _ = ACMD_Core(signal_1d, fs, max_components=5, tol=1e-4)
    
    # 4. 基于生理呼吸频段自适应能量重构
    reconstructed = base.reconstruct_top3_components_by_energy(components, fs)
    return reconstructed


# ================= 标准输出重定向模块 =================
class TextRedirector:
    """用于捕获 stdout 的 print 语句，并显示在 UI 日志面板中"""
    def __init__(self, write_func):
        self.write_func = write_func

    def write(self, text):
        if text.strip():
            self.write_func(text.strip())

    def flush(self):
        pass


# ================= GUI 主窗口类 =================
class MainTestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("呼吸算法滑窗分析测试工具 (VMD/EMD/ACMD)")
        self.resize(1500, 950)

        # 核心数据缓存
        self.file_path = None
        self.raw_times = None
        self.raw_frames = None
        self.clean_times = None
        self.clean_frames = None
        self.fs = 10.0
        
        # 滑窗结果缓存
        self.sliding_results = []
        
        # 原有 stdout 保存
        self.stdout_bak = sys.stdout

        self.setup_ui()
        
        # 挂载 stdout 重定向
        sys.stdout = TextRedirector(self.log_from_stdout)

    def setup_ui(self):
        # 采用深色极简风格设计
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1e;
            }
            QGroupBox {
                background-color: #24242b;
                border: 1px solid #3f3f46;
                border-radius: 8px;
                margin-top: 10px;
                color: #e4e4e7;
                font-weight: bold;
                font-size: 13px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                left: 10px;
            }
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #60a5fa;
            }
            QPushButton:pressed {
                background-color: #2563eb;
            }
            QPushButton:disabled {
                background-color: #4b5563;
                color: #9ca3af;
            }
            QLabel {
                color: #e4e4e7;
                font-size: 13px;
            }
            QComboBox {
                background-color: #27272a;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 5px;
                color: #f4f4f5;
                font-size: 13px;
            }
            QComboBox:disabled {
                background-color: #1f1f23;
                color: #71717a;
            }
            QLineEdit {
                background-color: #27272a;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 5px;
                color: #f4f4f5;
                font-size: 13px;
            }
            QCheckBox {
                color: #e4e4e7;
                font-size: 13px;
            }
            QTextEdit {
                background-color: #121214;
                color: #10b981;
                border: 1px solid #3f3f46;
                border-radius: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
            }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ================= 左侧控制面板 =================
        control_panel = QVBoxLayout()
        control_panel.setSpacing(15)

        # 1. 数据导入与清洗
        group_load = QGroupBox("1. 数据加载与清洗")
        layout_load = QVBoxLayout(group_load)
        self.btn_select_file = QPushButton("选择压力 TXT 文件")
        self.btn_preprocess = QPushButton("一键加载与清洗数据")
        self.btn_preprocess.setEnabled(False)
        layout_load.addWidget(self.btn_select_file)
        layout_load.addWidget(self.btn_preprocess)
        control_panel.addWidget(group_load)

        # 2. 算法与频率方法配置
        group_algo = QGroupBox("2. 算法与参数配置")
        layout_algo = QVBoxLayout(group_algo)
        
        self.algo_selector = QComboBox()
        self.algo_selector.addItems(["VMD", "EMD", "ACMD"])
        
        self.bpm_method_selector = QComboBox()
        self.bpm_method_selector.addItems(["FPR (特征点法)", "Peak (常规波峰法)"])

        layout_algo.addWidget(QLabel("核心提取算法:"))
        layout_algo.addWidget(self.algo_selector)
        layout_algo.addWidget(QLabel("呼吸频率 (BPM) 测量方法:"))
        layout_algo.addWidget(self.bpm_method_selector)
        control_panel.addWidget(group_algo)

        # 3. 滑窗配置 (Sliding Window Settings)
        group_slide = QGroupBox("3. 滑动窗口设置 (分帧)")
        layout_slide = QVBoxLayout(group_slide)
        
        self.cb_sliding_window = QCheckBox("启用滑动窗口分析")
        self.cb_sliding_window.setChecked(True)  # 默认开启滑窗

        # 窗口大小输入
        layout_w = QHBoxLayout()
        layout_w.addWidget(QLabel("窗口大小 (秒):"))
        self.win_size_input = QLineEdit("30")
        self.win_size_input.setAlignment(Qt.AlignCenter)
        layout_w.addWidget(self.win_size_input)
        
        # 步长输入
        layout_s = QHBoxLayout()
        layout_s.addWidget(QLabel("窗口步长 (秒):"))
        self.step_size_input = QLineEdit("5")
        self.step_size_input.setAlignment(Qt.AlignCenter)
        layout_s.addWidget(self.step_size_input)

        self.cb_align_phase = QCheckBox("自动对齐重叠区域相位")
        self.cb_align_phase.setChecked(True)
        self.cb_align_time = QCheckBox("对齐波形至相对时间轴 (0s起)")
        self.cb_align_time.setChecked(False)

        layout_slide.addWidget(self.cb_sliding_window)
        layout_slide.addLayout(layout_w)
        layout_slide.addLayout(layout_s)
        layout_slide.addWidget(self.cb_align_phase)
        layout_slide.addWidget(self.cb_align_time)
        control_panel.addWidget(group_slide)

        # 4. 执行分析与联动查看
        group_run = QGroupBox("4. 执行与查看")
        layout_run = QVBoxLayout(group_run)
        
        self.btn_analyze = QPushButton("开始提取呼吸")
        self.btn_analyze.setEnabled(False)
        
        self.combo_window_select = QComboBox()
        self.combo_window_select.setEnabled(False)
        self.combo_window_select.addItem("等待滑窗分析完成...")

        layout_run.addWidget(self.btn_analyze)
        layout_run.addWidget(QLabel("查看指定时间窗波形:"))
        layout_run.addWidget(self.combo_window_select)
        control_panel.addWidget(group_run)

        # 实时日志及状态
        self.status_label = QLabel("状态: 请选择数据文件")
        self.status_label.setStyleSheet("color: #60a5fa; font-weight: bold;")
        self.status_label.setWordWrap(True)
        control_panel.addWidget(self.status_label)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        control_panel.addWidget(QLabel("算法输出日志:"))
        control_panel.addWidget(self.log_edit)

        main_layout.addLayout(control_panel, 1)

        # ================= 右侧绘图区 =================
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)

        self.figure = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        main_layout.addWidget(plot_container, 3)

        # ================= 槽函数绑定 =================
        self.btn_select_file.clicked.connect(self.select_file)
        self.btn_preprocess.clicked.connect(self.preprocess_data)
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.combo_window_select.currentIndexChanged.connect(self.on_window_selection_changed)
        self.cb_align_time.stateChanged.connect(self.plot_sliding_window_results)

    # ================= 逻辑方法实现 =================
    
    def log(self, text):
        self.log_edit.append(f"<b>[SYSTEM]</b> {text}")
        QApplication.processEvents()

    def log_from_stdout(self, text):
        self.log_edit.append(text)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择压力数据", "", "Text Files (*.txt)")
        if path:
            self.file_path = path
            self.status_label.setText(f"已选择文件: {os.path.basename(path)}")
            self.btn_preprocess.setEnabled(True)
            self.log(f"成功选择数据文件: {path}")

    def preprocess_data(self):
        if not self.file_path: return
        self.log("正在解析数据文件，请稍候...")
        try:
            self.raw_times, self.raw_frames = data_loader.load_pressure_txt(self.file_path)
            if self.raw_frames is None or len(self.raw_frames) == 0:
                raise ValueError("数据加载为空，请检查文件格式。")
                
            self.log(f"成功导入: {len(self.raw_frames)} 帧原始数据。正在剔除异常波动坏帧...")
            
            # 清洗底噪及坏帧
            self.clean_times, self.clean_frames = preprocess.clean_dataset(self.raw_times, self.raw_frames)
            
            self.status_label.setText(f"清洗完成。有效帧数: {len(self.clean_frames)}")
            self.log(f"清洗完成。剔除坏帧后，共保留 {len(self.clean_frames)} 帧有效呼吸数据。")
            
            # 显示清洗后的空间均值趋势
            self.plot_spatial_mean(self.clean_frames, "受力点空间平均趋势 (清洗后)")
            self.btn_analyze.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "数据处理错误", f"解析失败:\n{str(e)}")
            self.log(f"数据解析失败: {str(e)}")

    def run_analysis(self):
        if self.clean_frames is None: return
        
        method = self.algo_selector.currentText()
        use_sliding = self.cb_sliding_window.isChecked()
        
        self.log_edit.clear()
        
        if use_sliding:
            self.run_sliding_analysis(method)
        else:
            self.run_normal_analysis(method)

    def run_normal_analysis(self, method):
        self.status_label.setText(f"正在进行全局分析 ({method})...")
        self.log(f"执行全局分析模式，算法: {method}")
        self.combo_window_select.clear()
        self.combo_window_select.setEnabled(False)
        self.combo_window_select.addItem("全局模式无需选窗")
        
        try:
            # 运行核心提取
            if method == "EMD":
                raw_breath = algorithms.extract_emd(self.clean_frames, self.fs)
            elif method == "VMD":
                raw_breath = algorithms.extract_vmd(self.clean_frames, self.fs)
            elif method == "ACMD":
                raw_breath = extract_acmd(self.clean_frames, self.fs)
                
            # 结果平滑与开头过渡切除 (防止坐姿初期影响)
            offset = min(100, len(raw_breath) // 4)
            processed = raw_breath[offset:]
            smoothed_breath = base.smooth_respiration_signal(processed)
            
            # 计算 BPM
            if self.bpm_method_selector.currentText() == "FPR (特征点法)":
                bpm = base.calculate_bpm_fpr(smoothed_breath, self.fs)
            else:
                bpm = base.calculate_bpm(smoothed_breath, self.fs)
                
            self.status_label.setText(f"全局分析完成！呼吸率: {bpm:.1f} BPM")
            self.plot_single_result(smoothed_breath, bpm, method)
            
        except Exception as e:
            QMessageBox.critical(self, "算法异常", f"全局分析失败:\n{str(e)}")
            self.log(f"分析失败: {str(e)}")

    def run_sliding_analysis(self, method):
        try:
            win_size_sec = float(self.win_size_input.text())
            step_size_sec = float(self.step_size_input.text())
        except ValueError:
            QMessageBox.warning(self, "参数错误", "窗口大小与步长必须为合法的数字。")
            return
            
        win_size_frames = int(win_size_sec * self.fs)
        step_size_frames = int(step_size_sec * self.fs)
        
        total_frames = len(self.clean_frames)
        if total_frames < win_size_frames:
            QMessageBox.warning(self, "数据不足", f"有效帧数 ({total_frames}) 不足滑窗长度 ({win_size_frames} 帧，{win_size_sec}秒)。已自动回退到全局分析。")
            self.cb_sliding_window.setChecked(False)
            self.run_normal_analysis(method)
            return
            
        # 划分滑动窗口区间
        windows = []
        start_idx = 0
        while start_idx + win_size_frames <= total_frames:
            end_idx = start_idx + win_size_frames
            windows.append((start_idx, end_idx))
            start_idx += step_size_frames
            
        self.sliding_results = []
        
        # 临时解绑槽函数，防止 addItem 触发 IndexChanged
        self.combo_window_select.blockSignals(True)
        self.combo_window_select.clear()
        
        self.log(f"开始滑动窗口分帧分析：算法={method}，窗口={win_size_sec}s，步长={step_size_sec}s，共 {len(windows)} 个分帧...")
        
        for idx, (start, end) in enumerate(windows):
            self.status_label.setText(f"正在解算分帧 {idx+1}/{len(windows)} ({start/self.fs:.1f}s - {end/self.fs:.1f}s)...")
            QApplication.processEvents()
            
            # 切片 3D frames 矩阵进行处理
            window_frames = self.clean_frames[start:end]
            
            # 调用对应算法提取
            if method == "EMD":
                raw_breath = extract_emd(window_frames, self.fs)
            elif method == "VMD":
                raw_breath = extract_vmd(window_frames, self.fs)
            elif method == "ACMD":
                raw_breath = extract_acmd(window_frames, self.fs)
                
            # 滑窗内部呼吸信号平滑（不需要再切除 100 帧）
            smoothed_breath = base.smooth_respiration_signal(raw_breath)
            
            # === 相位符号自动对齐逻辑 ===
            if self.cb_align_phase.isChecked() and idx > 0:
                overlap_len = win_size_frames - step_size_frames
                if overlap_len > 0:
                    # 前一窗口尾部重叠区域
                    overlap_prev = self.sliding_results[idx - 1]['wave'][-overlap_len:]
                    # 当前窗口头部重叠区域
                    overlap_curr = smoothed_breath[:overlap_len]
                    
                    # 互相关点积判定
                    corr = np.dot(overlap_prev, overlap_curr)
                    if corr < 0:
                        smoothed_breath = -smoothed_breath
                        self.log(f"   -> 分帧 {idx+1} 检测到相位反转，已自动取反对齐波形。")
            
            # 计算当前窗口的呼吸频率
            if self.bpm_method_selector.currentText() == "FPR (特征点法)":
                bpm = base.calculate_bpm_fpr(smoothed_breath, self.fs)
            else:
                bpm = base.calculate_bpm(smoothed_breath, self.fs)
                
            start_time = start / self.fs
            end_time = end / self.fs
            center_time = (start_time + end_time) / 2.0
            
            self.sliding_results.append({
                'window_idx': idx,
                'start_time': start_time,
                'end_time': end_time,
                'center_time': center_time,
                'wave': smoothed_breath,
                'bpm': bpm
            })
            
            self.combo_window_select.addItem(f"分帧 {idx+1}: {start_time:.1f}s - {end_time:.1f}s (BPM: {bpm:.1f})")
            
        # 在最末尾添加完整拼接波形的选项
        self.combo_window_select.addItem("完整拼接波形 (全段融合显示)")
            
        self.combo_window_select.blockSignals(False)
        self.combo_window_select.setEnabled(True)
        
        self.status_label.setText(f"分帧分析完成！共处理 {len(windows)} 个滑动区间。")
        self.log("所有滑动窗口呼吸提取完毕！已生成联动视图。")
        
        # 默认高亮第一个窗口
        self.combo_window_select.setCurrentIndex(0)
        self.plot_sliding_window_results()

    def on_window_selection_changed(self, idx):
        if not self.cb_sliding_window.isChecked() or idx < 0:
            return
        self.plot_sliding_window_results()

    def plot_spatial_mean(self, frames, title):
        self.figure.clear()
        self.figure.patch.set_facecolor('#1a1a1e')
        
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#1e1e24')
        
        avg_trend = []
        threshold = 100
        for f in frames:
            active_points = f[f > threshold]
            if active_points.size > 0:
                avg_trend.append(np.mean(active_points))
            else:
                non_zero = f[f > 35]
                avg_trend.append(np.mean(non_zero) if non_zero.size > 0 else 0)
                
        time_axis = np.arange(len(frames)) / self.fs
        
        ax.plot(time_axis, avg_trend, color='#3b82f6', linewidth=1.5)
        ax.set_title(f"{title} (阈值 > {threshold})", color='#e4e4e7', fontsize=14, fontweight='bold')
        ax.set_xlabel("时间 (秒)", color='#e4e4e7')
        ax.set_ylabel("受力点平均 ADC 强度", color='#e4e4e7')
        ax.tick_params(colors='#a1a1aa')
        ax.grid(True, color='#3f3f46', linestyle='--', alpha=0.5)
        
        for spine in ax.spines.values():
            spine.set_color('#3f3f46')
            
        self.canvas.draw()

    def plot_single_result(self, signal, bpm, method):
        self.figure.clear()
        self.figure.patch.set_facecolor('#1a1a1e')
        
        # 1. 绘制上半部：全局呼吸波形 (211)
        ax_wave = self.figure.add_subplot(211)
        ax_wave.set_facecolor('#1e1e24')
        
        time_axis = np.arange(len(signal)) / self.fs
        
        ax_wave.plot(time_axis, signal, color='#ef4444', linewidth=2, label='提取的呼吸分量')
        ax_wave.set_title(f"全局呼吸波形 ({method}) | 呼吸率: {bpm:.1f} BPM", color='#e4e4e7', fontsize=12, fontweight='bold')
        ax_wave.set_xlabel("时间 (秒)", color='#e4e4e7')
        ax_wave.set_ylabel("归一化强度", color='#e4e4e7')
        ax_wave.tick_params(colors='#a1a1aa')
        ax_wave.grid(True, color='#3f3f46', linestyle='--', alpha=0.5)
        ax_wave.legend(facecolor='#24242b', edgecolor='#3f3f46', labelcolor='#e4e4e7')
        
        for spine in ax_wave.spines.values():
            spine.set_color('#3f3f46')
            
        # 2. 绘制下半部：全局功率谱密度 PSD (212)
        ax_psd = self.figure.add_subplot(212)
        ax_psd.set_facecolor('#1e1e24')
        
        from scipy.signal import welch
        nperseg = min(len(signal), 256)
        f, psd = welch(signal, self.fs, nperseg=nperseg)
        
        ax_psd.plot(f, psd, color='#f59e0b', linewidth=2, label='功率谱密度 (PSD)')
        ax_psd.set_xlim(0, 1.5)
        ax_psd.set_title("全局呼吸信号功率谱密度 (PSD) 响应", color='#e4e4e7', fontsize=12, fontweight='bold')
        ax_psd.set_xlabel("频率 (Hz)", color='#e4e4e7')
        ax_psd.set_ylabel("功率谱密度", color='#e4e4e7')
        ax_psd.tick_params(colors='#a1a1aa')
        ax_psd.grid(True, color='#3f3f46', linestyle='--', alpha=0.5)
        
        # 寻找呼吸频段内最强谱峰并标记
        valid_mask = (f >= 0.1) & (f <= 0.5)
        if np.any(valid_mask):
            f_roi = f[valid_mask]
            psd_roi = psd[valid_mask]
            peak_idx = np.argmax(psd_roi)
            peak_freq = f_roi[peak_idx]
            peak_val = psd_roi[peak_idx]
            peak_bpm = peak_freq * 60.0
            
            ax_psd.scatter(peak_freq, peak_val, color='red', s=45, zorder=5)
            ax_psd.axvline(x=peak_freq, color='red', linestyle='--', alpha=0.5)
            ax_psd.text(peak_freq + 0.02, peak_val * 0.8, f"主峰: {peak_bpm:.1f} BPM\n({peak_freq:.3f} Hz)", color='red', fontweight='bold', fontsize=9)
            
        for spine in ax_psd.spines.values():
            spine.set_color('#3f3f46')
            
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_sliding_window_results(self):
        if not self.sliding_results: return
        
        self.figure.clear()
        self.figure.patch.set_facecolor('#1a1a1e')
        
        current_idx = self.combo_window_select.currentIndex()
        if current_idx < 0: current_idx = 0
        
        num_windows = len(self.sliding_results)
        is_merged_view = (current_idx == num_windows)
        
        # 1. 上子图: BPM 趋势折线图 (311)
        ax_trend = self.figure.add_subplot(311)
        ax_trend.set_facecolor('#1e1e24')
        
        times_sec = [r['center_time'] for r in self.sliding_results]
        bpms = [r['bpm'] for r in self.sliding_results]
        
        ax_trend.plot(times_sec, bpms, color='#3b82f6', marker='o', markersize=6, linewidth=2, label='BPM 趋势')
        
        if not is_merged_view:
            # 高亮当前选中的滑窗
            selected_time = times_sec[current_idx]
            selected_bpm = bpms[current_idx]
            ax_trend.scatter(selected_time, selected_bpm, color='#ef4444', s=120, zorder=5, label='当前选中窗口')
            ax_trend.axvline(x=selected_time, color='#ef4444', linestyle='--', alpha=0.7)
        else:
            # 融合视图：在趋势图上以阴影或高亮提示全局
            ax_trend.axhspan(min(bpms) - 1, max(bpms) + 1, color='#ef4444', alpha=0.08, label='全局融合段')
            
        ax_trend.set_title("呼吸率 (BPM) 变化趋势 (时间轴为窗口中心时间)", color='#e4e4e7', fontsize=11, fontweight='bold')
        ax_trend.set_xlabel("时间 (秒)", color='#e4e4e7')
        ax_trend.set_ylabel("呼吸率 (BPM)", color='#e4e4e7')
        ax_trend.tick_params(colors='#a1a1aa')
        ax_trend.grid(True, color='#3f3f46', linestyle='--', alpha=0.5)
        ax_trend.legend(facecolor='#24242b', edgecolor='#3f3f46', labelcolor='#e4e4e7')
        
        for spine in ax_trend.spines.values():
            spine.set_color('#3f3f46')
            
        # 2. 中子图: 呼吸波形 (明细或全段融合拼接) (312)
        ax_wave = self.figure.add_subplot(312)
        ax_wave.set_facecolor('#1e1e24')
        
        if is_merged_view:
            # 融合拼接逻辑：使用 Windowed Overlap-Add (加窗重叠相加平均) 融合成一个波形
            total_frames = len(self.clean_frames)
            wave = np.zeros(total_frames)
            weights = np.zeros(total_frames)
            
            for r in self.sliding_results:
                start_f = int(r['start_time'] * self.fs)
                end_f = int(r['end_time'] * self.fs)
                L = min(len(r['wave']), end_f - start_f)
                if L <= 0: continue
                
                # 构建平滑余弦渐变窗 (Tukey-like window)
                w = np.ones(L)
                taper_ratio = 0.15
                taper_len = int(L * taper_ratio)
                if taper_len > 1:
                    taper = 0.05 + 0.95 * (0.5 * (1.0 - np.cos(np.pi * np.arange(taper_len) / (taper_len - 1))))
                    w[:taper_len] = taper
                    w[-taper_len:] = taper[::-1]
                
                wave[start_f : start_f + L] += r['wave'][:L] * w
                weights[start_f : start_f + L] += w
                
            valid_mask = weights > 1e-5
            wave[valid_mask] /= weights[valid_mask]
            
            time_axis = np.arange(total_frames) / self.fs
            
            ax_wave.plot(time_axis, wave, color='#ec4899', linewidth=1.5, label='滑窗加窗融合拼接波形')
            ax_wave.set_title("完整波形图 (全段滑窗加窗平滑融合显示)", color='#e4e4e7', fontsize=11, fontweight='bold')
            ax_wave.set_xlabel("时间 (秒)", color='#e4e4e7')
        else:
            selected_res = self.sliding_results[current_idx]
            wave = selected_res['wave']
            start_time = selected_res['start_time']
            end_time = selected_res['end_time']
            bpm = selected_res['bpm']
            
            # 判断是否需要对齐到相对时间轴
            if self.cb_align_time.isChecked():
                time_axis = np.linspace(0, end_time - start_time, len(wave))
                ax_wave.plot(time_axis, wave, color='#10b981', linewidth=2, label='提取的呼吸分量')
                ax_wave.set_title(f"分帧明细 (对齐显示): 0.0s - {end_time - start_time:.1f}s | 对应呼吸率: {bpm:.1f} BPM (实际区间: {start_time:.1f}s - {end_time:.1f}s)", color='#e4e4e7', fontsize=11, fontweight='bold')
                ax_wave.set_xlabel("相对时间 (秒, 0s起)", color='#e4e4e7')
            else:
                time_axis = np.linspace(start_time, end_time, len(wave))
                ax_wave.plot(time_axis, wave, color='#10b981', linewidth=2, label='提取的呼吸分量')
                ax_wave.set_title(f"分帧明细: {start_time:.1f}s - {end_time:.1f}s | 对应呼吸率: {bpm:.1f} BPM", color='#e4e4e7', fontsize=11, fontweight='bold')
                ax_wave.set_xlabel("绝对时间 (秒)", color='#e4e4e7')
                
        ax_wave.set_ylabel("幅值 (归一化)", color='#e4e4e7')
        ax_wave.tick_params(colors='#a1a1aa')
        ax_wave.grid(True, color='#3f3f46', linestyle='--', alpha=0.5)
        ax_wave.legend(facecolor='#24242b', edgecolor='#3f3f46', labelcolor='#e4e4e7')
        
        for spine in ax_wave.spines.values():
            spine.set_color('#3f3f46')
            
        # 3. 下子图: 当前分帧的功率谱密度 PSD (313)
        ax_psd = self.figure.add_subplot(313)
        ax_psd.set_facecolor('#1e1e24')
        
        from scipy.signal import welch
        nperseg = min(len(wave), 256)
        f, psd = welch(wave, self.fs, nperseg=nperseg)
        
        ax_psd.plot(f, psd, color='#f59e0b', linewidth=2, label='功率谱密度 (PSD)')
        ax_psd.set_xlim(0, 1.5)
        
        if is_merged_view:
            ax_psd.set_title("全局融合呼吸信号功率谱密度 (PSD) 响应", color='#e4e4e7', fontsize=11, fontweight='bold')
        else:
            ax_psd.set_title(f"当前分帧呼吸信号功率谱密度 (PSD) 响应 | 对应呼吸率: {bpm:.1f} BPM", color='#e4e4e7', fontsize=11, fontweight='bold')
            
        ax_psd.set_xlabel("频率 (Hz)", color='#e4e4e7')
        ax_psd.set_ylabel("功率谱密度", color='#e4e4e7')
        ax_psd.tick_params(colors='#a1a1aa')
        ax_psd.grid(True, color='#3f3f46', linestyle='--', alpha=0.5)
        
        # 寻找呼吸频段内最强谱峰并标记
        valid_mask = (f >= 0.1) & (f <= 0.5)
        if np.any(valid_mask):
            f_roi = f[valid_mask]
            psd_roi = psd[valid_mask]
            peak_idx = np.argmax(psd_roi)
            peak_freq = f_roi[peak_idx]
            peak_val = psd_roi[peak_idx]
            peak_bpm = peak_freq * 60.0
            
            ax_psd.scatter(peak_freq, peak_val, color='red', s=45, zorder=5)
            ax_psd.axvline(x=peak_freq, color='red', linestyle='--', alpha=0.5)
            ax_psd.text(peak_freq + 0.02, peak_val * 0.8, f"主峰: {peak_bpm:.1f} BPM\n({peak_freq:.3f} Hz)", color='red', fontweight='bold', fontsize=9)
            
        for spine in ax_psd.spines.values():
            spine.set_color('#3f3f46')
            
        self.figure.tight_layout()
        self.canvas.draw()

    def closeEvent(self, event):
        # 窗口关闭时还原系统标准输出
        sys.stdout = self.stdout_bak
        super().closeEvent(event)


if __name__ == "__main__":
    # 配置中文字体支持
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    
    app = QApplication(sys.argv)
    window = MainTestWindow()
    window.show()
    sys.exit(app.exec_())
