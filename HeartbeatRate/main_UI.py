import sys
import os
import numpy as np

# 修复 conda 环境下 PyQt5 DLL 找不到的问题
# 检测是否在 conda 环境中，并将 Library\bin 加入 PATH
_conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)
_qt_dll_path = os.path.join(_conda_prefix, 'Library', 'bin')
if os.path.isdir(_qt_dll_path) and _qt_dll_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _qt_dll_path + os.pathsep + os.environ.get('PATH', '')

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

# --- 动态添加当前目录到 sys.path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入底层核心数据处理与算法包
import data_loader
import preprocess
import algorithms
from algorithms import base


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
class HeartbeatSlidingWindowUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("心跳算法滑窗分析测试工具 (ACMD/VMD/EMD)")
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
        self.algo_selector.addItems(["ACMD", "VMD", "EMD", "VME"])
        
        self.bpm_method_selector = QComboBox()
        self.bpm_method_selector.addItems(["FPR (特征点法)", "Peak (常规波峰法)"])

        layout_algo.addWidget(QLabel("心跳提取算法:"))
        layout_algo.addWidget(self.algo_selector)
        layout_algo.addWidget(QLabel("心率 (BPM) 测量方法:"))
        layout_algo.addWidget(self.bpm_method_selector)
        control_panel.addWidget(group_algo)

        # 3. 滑窗配置 (Sliding Window Settings)
        group_slide = QGroupBox("3. 滑动窗口设置 (分帧)")
        layout_slide = QVBoxLayout(group_slide)
        
        self.cb_sliding_window = QCheckBox("启用滑动窗口分析")
        self.cb_sliding_window.setChecked(True)  # 默认开启滑窗

        # 窗口大小输入（心跳快，因此滑窗建议 15 秒）
        layout_w = QHBoxLayout()
        layout_w.addWidget(QLabel("窗口大小 (秒):"))
        self.win_size_input = QLineEdit("15")
        self.win_size_input.setAlignment(Qt.AlignCenter)
        layout_w.addWidget(self.win_size_input)
        
        # 步长输入
        layout_s = QHBoxLayout()
        layout_s.addWidget(QLabel("窗口步长 (秒):"))
        self.step_size_input = QLineEdit("3")
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
        
        self.btn_analyze = QPushButton("开始提取心跳")
        self.btn_analyze.setEnabled(False)
        
        self.combo_window_select = QComboBox()
        self.combo_window_select.setEnabled(False)
        self.combo_window_select.addItem("等待滑窗分析完成...")

        layout_run.addWidget(self.btn_analyze)
        layout_run.addWidget(QLabel("查看指定时间窗波形:"))
        layout_run.addWidget(self.combo_window_select)
        control_panel.addWidget(group_run)

        # 实时状态显示
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
                
            self.log(f"成功导入: {len(self.raw_frames)} 帧原始数据。正在进行帧去噪及清洗...")
            
            # 清洗底噪及剔除异常帧
            self.clean_times, self.clean_frames = preprocess.clean_dataset(self.raw_times, self.raw_frames)
            
            self.status_label.setText(f"清洗完成。有效帧数: {len(self.clean_frames)}")
            self.log(f"预处理清洗完成。剔除坏帧后，共保留 {len(self.clean_frames)} 帧有效数据。")
            
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
            # 运行心跳提取算法
            if method == "EMD":
                raw_hb = algorithms.extract_emd(self.clean_frames, self.fs)
            elif method == "VMD":
                raw_hb = algorithms.extract_vmd(self.clean_frames, self.fs)
            elif method == "ACMD":
                raw_hb = algorithms.extract_acmd(self.clean_frames, self.fs)
            elif method == "VME":
                raw_hb = algorithms.extract_vme(self.clean_frames, self.fs)
                
            # 移除硬编码的开头切除 (因已经在 preprocess 中切除首尾 20s)
            processed = raw_hb
            smoothed_hb = base.smooth_heartbeat_signal(processed)
            
            # 计算心率 (BPM)
            if self.bpm_method_selector.currentText() == "FPR (特征点法)":
                bpm = base.calculate_bpm_fpr(smoothed_hb, self.fs)
            else:
                bpm = base.calculate_bpm(smoothed_hb, self.fs)
                
            self.status_label.setText(f"全局心脉分析完成！心率: {bpm:.1f} BPM")
            self.plot_single_result(smoothed_hb, bpm, method)
            
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
        
        # 临时解绑槽函数，防止重复触发
        self.combo_window_select.blockSignals(True)
        self.combo_window_select.clear()
        
        self.log(f"开始心率滑窗分帧解算：算法={method}，窗口={win_size_sec}s，步长={step_size_sec}s，共 {len(windows)} 个分帧段...")
        
        for idx, (start, end) in enumerate(windows):
            self.status_label.setText(f"正在解算分帧 {idx+1}/{len(windows)} ({start/self.fs:.1f}s - {end/self.fs:.1f}s)...")
            QApplication.processEvents()
            
            # 切片 3D frames 矩阵进行处理
            window_frames = self.clean_frames[start:end]
            
            # 运行对应的心跳提取算法
            if method == "EMD":
                raw_hb = algorithms.extract_emd(window_frames, self.fs)
            elif method == "VMD":
                raw_hb = algorithms.extract_vmd(window_frames, self.fs)
            elif method == "ACMD":
                raw_hb = algorithms.extract_acmd(window_frames, self.fs)
            elif method == "VME":
                raw_hb = algorithms.extract_vme(window_frames, self.fs)
                
            # 心搏滤波平滑
            smoothed_hb = base.smooth_heartbeat_signal(raw_hb)
            
            # === 相位符号自适应对齐 ===
            if self.cb_align_phase.isChecked() and idx > 0:
                overlap_len = win_size_frames - step_size_frames
                if overlap_len > 0:
                    overlap_prev = self.sliding_results[idx - 1]['wave'][-overlap_len:]
                    overlap_curr = smoothed_hb[:overlap_len]
                    corr = np.dot(overlap_prev, overlap_curr)
                    if corr < 0:
                        smoothed_hb = -smoothed_hb
                        self.log(f"   -> 分帧 {idx+1} 检测到相位反向，已自动翻转对齐。")
            
            # 计算当前窗口的心率 (BPM)
            if self.bpm_method_selector.currentText() == "FPR (特征点法)":
                bpm = base.calculate_bpm_fpr(smoothed_hb, self.fs)
            else:
                bpm = base.calculate_bpm(smoothed_hb, self.fs)
                
            start_time = start / self.fs
            end_time = end / self.fs
            center_time = (start_time + end_time) / 2.0
            
            self.sliding_results.append({
                'window_idx': idx,
                'start_time': start_time,
                'end_time': end_time,
                'center_time': center_time,
                'wave': smoothed_hb,
                'bpm': bpm
            })
            
            self.combo_window_select.addItem(f"分帧 {idx+1}: {start_time:.1f}s - {end_time:.1f}s (BPM: {bpm:.1f})")
            
        # 在最末尾添加完整拼接波形的选项
        self.combo_window_select.addItem("完整拼接波形 (全段融合显示)")
            
        self.combo_window_select.blockSignals(False)
        self.combo_window_select.setEnabled(True)
        
        self.status_label.setText(f"分帧分析完成！共处理 {len(windows)} 个滑窗区间。")
        self.log("所有滑动窗口心脉数据提取完毕！已生成联动视图。")
        
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
        
        # 1. 绘制上半部：全局心搏波形 (211)
        ax_wave = self.figure.add_subplot(211)
        ax_wave.set_facecolor('#1e1e24')
        
        time_axis = np.arange(len(signal)) / self.fs
        
        ax_wave.plot(time_axis, signal, color='#ef4444', linewidth=2, label='提取的 BCG 心跳分量')
        
        # 标记波峰
        min_dist = int(self.fs * 0.4)
        peaks, _ = plt.mlab.find_peaks(signal, min_dist) if hasattr(plt.mlab, 'find_peaks') else ([], None)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(signal, distance=min_dist, prominence=(np.max(signal) - np.min(signal)) * 0.1)
        ax_wave.scatter(time_axis[peaks], signal[peaks], color='#3b82f6', s=45, zorder=5, label='检测心脉峰值')
        
        ax_wave.set_title(f"全局心搏重构波形 ({method}) | 心率估计: {bpm:.1f} BPM", color='#e4e4e7', fontsize=12, fontweight='bold')
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
        
        ax_psd.plot(f, psd, color='#f59e0b', linewidth=2, label='振幅谱 (PSD)')
        ax_psd.axvspan(0.8, 2.2, color='green', alpha=0.1, label="心脉正常区间 (0.8-2.2 Hz)")
        ax_psd.set_xlim(0, 4.0)
        ax_psd.set_title("全局心跳信号幅值谱响应", color='#e4e4e7', fontsize=12, fontweight='bold')
        ax_psd.set_xlabel("频率 (Hz)", color='#e4e4e7')
        ax_psd.set_ylabel("幅值", color='#e4e4e7')
        ax_psd.tick_params(colors='#a1a1aa')
        ax_psd.grid(True, color='#3f3f46', linestyle='--', alpha=0.5)
        ax_psd.legend(facecolor='#24242b', edgecolor='#3f3f46', labelcolor='#e4e4e7')
        
        # 寻找心跳段最强谱峰并标记
        valid_mask = (f >= 0.8) & (f <= 2.2)
        if np.any(valid_mask):
            f_roi = f[valid_mask]
            psd_roi = psd[valid_mask]
            peak_idx = np.argmax(psd_roi)
            peak_freq = f_roi[peak_idx]
            peak_val = psd_roi[peak_idx]
            peak_bpm = peak_freq * 60.0
            
            ax_psd.scatter(peak_freq, peak_val, color='red', s=45, zorder=5)
            ax_psd.axvline(x=peak_freq, color='red', linestyle='--', alpha=0.5)
            ax_psd.text(peak_freq + 0.05, peak_val * 0.8, f"心率主谱峰: {peak_bpm:.1f} BPM\n({peak_freq:.2f} Hz)", color='red', fontweight='bold', fontsize=9)
            
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
        
        ax_trend.plot(times_sec, bpms, color='#3b82f6', marker='o', markersize=6, linewidth=2, label='心率趋势 (BPM)')
        
        if not is_merged_view:
            selected_time = times_sec[current_idx]
            selected_bpm = bpms[current_idx]
            ax_trend.scatter(selected_time, selected_bpm, color='#ef4444', s=120, zorder=5, label='当前选中时间窗')
            ax_trend.axvline(x=selected_time, color='#ef4444', linestyle='--', alpha=0.7)
        else:
            ax_trend.axhspan(min(bpms) - 2, max(bpms) + 2, color='#ef4444', alpha=0.08, label='全局融合范围')
            
        ax_trend.set_title("心率 (BPM) 变化趋势折线图", color='#e4e4e7', fontsize=11, fontweight='bold')
        ax_trend.set_xlabel("时间 (秒)", color='#e4e4e7')
        ax_trend.set_ylabel("心率 (BPM)", color='#e4e4e7')
        ax_trend.tick_params(colors='#a1a1aa')
        ax_trend.grid(True, color='#3f3f46', linestyle='--', alpha=0.5)
        ax_trend.legend(facecolor='#24242b', edgecolor='#3f3f46', labelcolor='#e4e4e7')
        
        for spine in ax_trend.spines.values():
            spine.set_color('#3f3f46')
            
        # 2. 中子图: 心跳波形 (明细或全段融合拼接) (312)
        ax_wave = self.figure.add_subplot(312)
        ax_wave.set_facecolor('#1e1e24')
        
        if is_merged_view:
            # 融合拼接逻辑
            total_frames = len(self.clean_frames)
            wave = np.zeros(total_frames)
            weights = np.zeros(total_frames)
            
            for r in self.sliding_results:
                start_f = int(r['start_time'] * self.fs)
                end_f = int(r['end_time'] * self.fs)
                L = min(len(r['wave']), end_f - start_f)
                if L <= 0: continue
                
                # 构建加窗渐变系数，确保交界平滑过度
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
            ax_wave.plot(time_axis, wave, color='#ec4899', linewidth=1.5, label='滑窗融合心脉波形')
            ax_wave.set_title("心搏完整波形图 (全段滑窗重叠相加平滑拼接)", color='#e4e4e7', fontsize=11, fontweight='bold')
            ax_wave.set_xlabel("时间 (秒)", color='#e4e4e7')
        else:
            selected_res = self.sliding_results[current_idx]
            wave = selected_res['wave']
            start_time = selected_res['start_time']
            end_time = selected_res['end_time']
            bpm = selected_res['bpm']
            
            # 标记波峰
            from scipy.signal import find_peaks
            min_dist = int(self.fs * 0.4)
            peaks, _ = find_peaks(wave, distance=min_dist, prominence=(np.max(wave) - np.min(wave)) * 0.1)
            
            if self.cb_align_time.isChecked():
                time_axis = np.linspace(0, end_time - start_time, len(wave))
                ax_wave.plot(time_axis, wave, color='#10b981', linewidth=2, label='提取的 J波心脉波')
                ax_wave.scatter(time_axis[peaks], wave[peaks], color='#3b82f6', s=45, zorder=5, label='检测波峰')
                ax_wave.set_title(f"分帧明细 (对齐显示): 0.0s - {end_time - start_time:.1f}s | 心率估算: {bpm:.1f} BPM (绝对区间: {start_time:.1f}s - {end_time:.1f}s)", color='#e4e4e7', fontsize=11, fontweight='bold')
                ax_wave.set_xlabel("相对时间 (秒, 0s起)", color='#e4e4e7')
            else:
                time_axis = np.linspace(start_time, end_time, len(wave))
                ax_wave.plot(time_axis, wave, color='#10b981', linewidth=2, label='提取的 J波心脉波')
                ax_wave.scatter(time_axis[peaks], wave[peaks], color='#3b82f6', s=45, zorder=5, label='检测波峰')
                ax_wave.set_title(f"分帧明细: {start_time:.1f}s - {end_time:.1f}s | 心率估算: {bpm:.1f} BPM", color='#e4e4e7', fontsize=11, fontweight='bold')
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
        
        ax_psd.plot(f, psd, color='#f59e0b', linewidth=2, label='幅值谱 (PSD)')
        ax_psd.axvspan(0.8, 2.2, color='green', alpha=0.1, label="心跳带 (0.8-2.2 Hz)")
        ax_psd.set_xlim(0, 4.0)
        
        if is_merged_view:
            ax_psd.set_title("拼接波形全局幅值谱响应", color='#e4e4e7', fontsize=11, fontweight='bold')
        else:
            ax_psd.set_title(f"当前分帧幅值谱响应 | 生理心率: {bpm:.1f} BPM", color='#e4e4e7', fontsize=11, fontweight='bold')
            
        ax_psd.set_xlabel("频率 (Hz)", color='#e4e4e7')
        ax_psd.set_ylabel("幅值", color='#e4e4e7')
        ax_psd.tick_params(colors='#a1a1aa')
        ax_psd.grid(True, color='#3f3f46', linestyle='--', alpha=0.5)
        ax_psd.legend(facecolor='#24242b', edgecolor='#3f3f46', labelcolor='#e4e4e7')
        
        # 寻找心跳频段内最强谱峰并标记
        valid_mask = (f >= 0.8) & (f <= 2.2)
        if np.any(valid_mask):
            f_roi = f[valid_mask]
            psd_roi = psd[valid_mask]
            peak_idx = np.argmax(psd_roi)
            peak_freq = f_roi[peak_idx]
            peak_val = psd_roi[peak_idx]
            peak_bpm = peak_freq * 60.0
            
            ax_psd.scatter(peak_freq, peak_val, color='red', s=45, zorder=5)
            ax_psd.axvline(x=peak_freq, color='red', linestyle='--', alpha=0.5)
            ax_psd.text(peak_freq + 0.05, peak_val * 0.8, f"心率主谱峰: {peak_bpm:.1f} BPM\n({peak_freq:.2f} Hz)", color='red', fontweight='bold', fontsize=9)
            
        for spine in ax_psd.spines.values():
            spine.set_color('#3f3f46')
            
        self.figure.tight_layout()
        self.canvas.draw()

    def keyPressEvent(self, event):
        # 如果滑窗下拉菜单未启用，或者有输入框处于焦点，则不响应快捷键以防止冲突
        if not self.combo_window_select.isEnabled():
            super().keyPressEvent(event)
            return

        # 检查当前是否有人脸/压力输入框等文字输入控件具有焦点
        focused_widget = QApplication.focusWidget()
        if isinstance(focused_widget, QLineEdit):
            super().keyPressEvent(event)
            return

        current_idx = self.combo_window_select.currentIndex()
        count = self.combo_window_select.count()
        
        if event.key() == Qt.Key_Left:
            if current_idx > 0:
                self.combo_window_select.setCurrentIndex(current_idx - 1)
                self.log(f"键盘快捷键：切换至上一分帧时间窗 -> {self.combo_window_select.currentText()}")
        elif event.key() == Qt.Key_Right:
            if current_idx < count - 1:
                self.combo_window_select.setCurrentIndex(current_idx + 1)
                self.log(f"键盘快捷键：切换至下一分帧时间窗 -> {self.combo_window_select.currentText()}")
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        # 还原系统输出
        sys.stdout = self.stdout_bak
        super().closeEvent(event)


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    
    app = QApplication(sys.argv)
    window = HeartbeatSlidingWindowUI()
    window.show()
    sys.exit(app.exec_())
