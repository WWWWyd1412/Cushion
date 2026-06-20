"""
Breath Extraction — 呼吸信号滑窗分析主界面 (PyQt5)

支持算法: EMD / VMD / AFD / VMD_FPR(MAPE) / GOA-VMD
滑窗参数: 窗口 250 帧 (25s), 步长 50 帧 (5s)
导航: 下拉框选择窗口 / 左右方向键切换窗口
显示: BPM 趋势 + 波形 + PSD (三子图)
"""

import sys
import os
import numpy as np

# 修复 conda 环境下 PyQt5 DLL 找不到的问题
_conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)
_qt_dll_path = os.path.join(_conda_prefix, 'Library', 'bin')
if os.path.isdir(_qt_dll_path) and _qt_dll_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _qt_dll_path + os.pathsep + os.environ.get('PATH', '')

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QMessageBox, QComboBox, QGroupBox, QLineEdit,
                             QCheckBox, QTextEdit, QProgressBar)
from PyQt5.QtCore import Qt

# 导入自定义模块
import data_loader
import preprocess
import algorithms
from algorithms import base


# ================= 标准输出重定向模块 =================
class TextRedirector:
    """捕获 stdout 的 print 语句，显示在 UI 日志面板中"""
    def __init__(self, write_func):
        self.write_func = write_func

    def write(self, text):
        if text.strip():
            self.write_func(text.strip())

    def flush(self):
        pass


# ================= GUI 主窗口 =================
class BreathSlidingWindowUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("呼吸信号滑窗分析系统 (EMD/VMD/AFD/VMD_FPR/GOA-VMD)")
        self.resize(1500, 950)

        # ---- 核心数据缓存 ----
        self.file_path = None
        self.raw_times = None
        self.raw_frames = None
        self.clean_times = None
        self.clean_frames = None
        self.fs = 10.0

        # ---- 滑窗结果缓存 ----
        self.sliding_results = []

        # ---- 保存原始 stdout ----
        self.stdout_bak = sys.stdout

        self.setup_ui()

        # 挂载 stdout 重定向
        sys.stdout = TextRedirector(self.log_from_stdout)

    # ================= UI 构建 =================
    def setup_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a1e; }
            QGroupBox {
                background-color: #24242b; border: 1px solid #3f3f46;
                border-radius: 8px; margin-top: 10px; color: #e4e4e7;
                font-weight: bold; font-size: 13px; padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; subcontrol-position: top left;
                padding: 0 5px; left: 10px;
            }
            QPushButton {
                background-color: #3b82f6; color: white; border: none;
                border-radius: 6px; padding: 8px 16px; font-weight: bold; font-size: 13px;
            }
            QPushButton:hover { background-color: #60a5fa; }
            QPushButton:pressed { background-color: #2563eb; }
            QPushButton:disabled { background-color: #4b5563; color: #9ca3af; }
            QLabel { color: #e4e4e7; font-size: 13px; }
            QComboBox {
                background-color: #27272a; border: 1px solid #3f3f46;
                border-radius: 6px; padding: 5px; color: #f4f4f5; font-size: 13px;
            }
            QComboBox:disabled { background-color: #1f1f23; color: #71717a; }
            QLineEdit {
                background-color: #27272a; border: 1px solid #3f3f46;
                border-radius: 6px; padding: 5px; color: #f4f4f5; font-size: 13px;
            }
            QCheckBox { color: #e4e4e7; font-size: 13px; }
            QTextEdit {
                background-color: #121214; color: #10b981;
                border: 1px solid #3f3f46; border-radius: 8px;
                font-family: 'Consolas', 'Courier New', monospace; font-size: 12px;
            }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ===== 左侧控制面板 =====
        control_panel = QVBoxLayout()
        control_panel.setSpacing(15)

        # 1. 数据加载与清洗
        group_load = QGroupBox("1. 数据加载与清洗")
        layout_load = QVBoxLayout(group_load)
        self.btn_select_file = QPushButton("选择压力 TXT 文件")
        self.btn_preprocess = QPushButton("一键加载与清洗数据")
        self.btn_preprocess.setEnabled(False)
        layout_load.addWidget(self.btn_select_file)
        layout_load.addWidget(self.btn_preprocess)
        control_panel.addWidget(group_load)

        # 2. 算法与参数配置
        group_algo = QGroupBox("2. 算法与参数配置")
        layout_algo = QVBoxLayout(group_algo)

        self.algo_selector = QComboBox()
        self.algo_selector.addItems(["EMD", "VMD", "AFD", "VMD_FPR", "SMVMD", "MVMD", "Multi-ROI ICA"])

        self.bpm_method_selector = QComboBox()
        self.bpm_method_selector.addItems(["Peak (常规波峰法)", "FPR (特征点法)"])

        layout_algo.addWidget(QLabel("呼吸提取算法:"))
        layout_algo.addWidget(self.algo_selector)
        layout_algo.addWidget(QLabel("呼吸率 (BPM) 测量方法:"))
        layout_algo.addWidget(self.bpm_method_selector)
        control_panel.addWidget(group_algo)

        # 3. 滑窗配置
        group_slide = QGroupBox("3. 滑动窗口设置")
        layout_slide = QVBoxLayout(group_slide)

        self.cb_sliding_window = QCheckBox("启用滑动窗口分析")
        self.cb_sliding_window.setChecked(True)

        layout_w = QHBoxLayout()
        layout_w.addWidget(QLabel("窗口大小 (帧):"))
        self.win_size_input = QLineEdit("250")
        self.win_size_input.setAlignment(Qt.AlignCenter)
        layout_w.addWidget(self.win_size_input)

        layout_s = QHBoxLayout()
        layout_s.addWidget(QLabel("窗口步长 (帧):"))
        self.step_size_input = QLineEdit("50")
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

        # 4. 执行与查看
        group_run = QGroupBox("4. 执行与查看")
        layout_run = QVBoxLayout(group_run)

        self.btn_analyze = QPushButton("开始提取呼吸信号")
        self.btn_analyze.setEnabled(False)

        self.combo_window_select = QComboBox()
        self.combo_window_select.setEnabled(False)
        self.combo_window_select.addItem("等待滑窗分析完成...")

        layout_run.addWidget(self.btn_analyze)
        layout_run.addWidget(QLabel("查看指定时间窗 (← → 键切换):"))
        layout_run.addWidget(self.combo_window_select)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("等待分析...")
        layout_run.addWidget(self.progress_bar)

        control_panel.addWidget(group_run)

        # 状态 & 日志
        self.status_label = QLabel("状态: 请选择数据文件")
        self.status_label.setStyleSheet("color: #60a5fa; font-weight: bold;")
        self.status_label.setWordWrap(True)
        control_panel.addWidget(self.status_label)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        control_panel.addWidget(QLabel("算法输出日志:"))
        control_panel.addWidget(self.log_edit)

        main_layout.addLayout(control_panel, 1)

        # ===== 右侧绘图区 =====
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)

        self.figure = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        main_layout.addWidget(plot_container, 3)

        # ===== 槽函数绑定 =====
        self.btn_select_file.clicked.connect(self.select_file)
        self.btn_preprocess.clicked.connect(self.preprocess_data)
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.combo_window_select.currentIndexChanged.connect(self.on_window_selection_changed)
        self.cb_align_time.stateChanged.connect(self.plot_sliding_window_results)

    # ================= 日志方法 =================
    def log(self, text):
        self.log_edit.append(f"<b>[SYSTEM]</b> {text}")
        QApplication.processEvents()

    def log_from_stdout(self, text):
        self.log_edit.append(text)

    # ================= 数据加载 & 清洗 =================
    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择压力数据", "", "Text Files (*.txt)")
        if path:
            self.file_path = path
            self.status_label.setText(f"已选择文件: {os.path.basename(path)}")
            self.btn_preprocess.setEnabled(True)
            self.log(f"成功选择数据文件: {path}")

    def preprocess_data(self):
        if not self.file_path:
            return
        self.log("正在解析数据文件，请稍候...")
        try:
            self.raw_times, self.raw_frames = data_loader.load_pressure_txt(self.file_path)
            if self.raw_frames is None or len(self.raw_frames) == 0:
                raise ValueError("数据加载为空，请检查文件格式。")

            self.log(f"成功导入: {len(self.raw_frames)} 帧原始数据。正在清洗...")

            self.clean_times, self.clean_frames = preprocess.clean_dataset(
                self.raw_times, self.raw_frames)

            self.status_label.setText(f"清洗完成。有效帧数: {len(self.clean_frames)}")
            self.log(f"清洗完成。剔除坏帧后，共保留 {len(self.clean_frames)} 帧有效数据。")

            self.plot_spatial_mean(self.clean_frames, "受力点空间平均趋势 (清洗后)")
            self.btn_analyze.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "数据处理错误", f"解析失败:\n{str(e)}")
            self.log(f"数据解析失败: {str(e)}")

    # ================= 分析调度 =================
    def run_analysis(self):
        if self.clean_frames is None:
            return

        method = self.algo_selector.currentText()
        use_sliding = self.cb_sliding_window.isChecked()

        self.log_edit.clear()

        if use_sliding:
            self.run_sliding_analysis(method)
        else:
            self.run_normal_analysis(method)

    # ================= 全局分析模式 =================
    def run_normal_analysis(self, method):
        self.status_label.setText(f"正在进行全局分析 ({method})...")
        self.log(f"执行全局分析模式，算法: {method}")
        self.combo_window_select.clear()
        self.combo_window_select.setEnabled(False)
        self.combo_window_select.addItem("全局模式无需选窗")

        # 进度条
        self.progress_bar.setMaximum(0)  # 不确定模式 (或 100)
        self.progress_bar.setFormat("正在分析...")
        self.btn_analyze.setEnabled(False)
        QApplication.processEvents()

        try:
            # GOA-VMD 子进度回调
            if method == "GOA-VMD":
                def _goa_progress(cur_iter, max_iter):
                    self.progress_bar.setMaximum(max_iter)
                    self.progress_bar.setValue(cur_iter)
                    self.progress_bar.setFormat(f"GOA 优化第 {cur_iter}/{max_iter} 代")
                    QApplication.processEvents()
            else:
                _goa_progress = None

            raw_breath = self._call_algorithm(method, self.clean_frames, self.fs,
                                               progress_callback=_goa_progress)

            offset = min(100, len(raw_breath) // 4)
            processed = raw_breath[offset:]
            smoothed_breath = base.smooth_respiration_signal(processed)

            bpm = self._calc_bpm(smoothed_breath, self.fs)

            self.status_label.setText(f"全局分析完成！呼吸率: {bpm:.1f} BPM")
            self.progress_bar.setMaximum(100)
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("分析完成!")
            self.btn_analyze.setEnabled(True)
            self.plot_single_result(smoothed_breath, bpm, method)

        except Exception as e:
            self.progress_bar.setFormat("分析失败")
            self.btn_analyze.setEnabled(True)
            QMessageBox.critical(self, "算法异常", f"全局分析失败:\n{str(e)}")
            self.log(f"分析失败: {str(e)}")

    # ================= 滑窗分析模式 =================
    def run_sliding_analysis(self, method):
        try:
            win_size_frames = int(self.win_size_input.text())
            step_size_frames = int(self.step_size_input.text())
        except ValueError:
            QMessageBox.warning(self, "参数错误", "窗口大小与步长必须为合法的数字。")
            return

        total_frames = len(self.clean_frames)
        if total_frames < win_size_frames:
            QMessageBox.warning(
                self, "数据不足",
                f"有效帧数 ({total_frames}) 不足滑窗长度 ({win_size_frames} 帧)。"
                f"已自动回退到全局分析。")
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

        self.combo_window_select.blockSignals(True)
        self.combo_window_select.clear()

        self.log(f"开始滑动窗口分帧分析：算法={method}，"
                 f"窗口={win_size_frames}帧({win_size_frames/self.fs:.1f}s)，"
                 f"步长={step_size_frames}帧({step_size_frames/self.fs:.1f}s)，"
                 f"共 {len(windows)} 个分帧...")

        # 初始化进度条
        self.progress_bar.setMaximum(len(windows))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("准备分析...")
        self.btn_analyze.setEnabled(False)

        for idx, (start, end) in enumerate(windows):
            # 更新窗口级进度
            self.progress_bar.setValue(idx)
            self.progress_bar.setFormat(
                f"分帧 {idx+1}/{len(windows)} "
                f"({start/self.fs:.1f}s - {end/self.fs:.1f}s)")
            self.status_label.setText(
                f"正在解算分帧 {idx+1}/{len(windows)} "
                f"({start/self.fs:.1f}s - {end/self.fs:.1f}s)...")
            QApplication.processEvents()

            # GOA-VMD 子进度回调
            if method == "GOA-VMD":
                def _goa_progress(cur_iter, max_iter):
                    self.progress_bar.setFormat(
                        f"分帧 {idx+1}/{len(windows)} — "
                        f"GOA 优化第 {cur_iter}/{max_iter} 代")
                    QApplication.processEvents()
            else:
                _goa_progress = None

            window_frames = self.clean_frames[start:end]
            raw_breath = self._call_algorithm(method, window_frames, self.fs,
                                               progress_callback=_goa_progress)
            smoothed_breath = base.smooth_respiration_signal(raw_breath)

            # 相位符号自动对齐
            if self.cb_align_phase.isChecked() and idx > 0:
                overlap_len = win_size_frames - step_size_frames
                if overlap_len > 0 and overlap_len < len(smoothed_breath):
                    overlap_prev = self.sliding_results[idx - 1]['wave'][-overlap_len:]
                    overlap_curr = smoothed_breath[:overlap_len]
                    corr = np.dot(overlap_prev, overlap_curr)
                    if corr < 0:
                        smoothed_breath = -smoothed_breath
                        self.log(f"   -> 分帧 {idx+1} 检测到相位反转，已自动取反对齐。")

            bpm = self._calc_bpm(smoothed_breath, self.fs)

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

            self.combo_window_select.addItem(
                f"分帧 {idx+1}: {start_time:.1f}s - {end_time:.1f}s (BPM: {bpm:.1f})")

        # 最后一项：完整拼接波形
        self.combo_window_select.addItem("完整拼接波形 (全段融合显示)")

        self.combo_window_select.blockSignals(False)
        self.combo_window_select.setEnabled(True)

        # 进度条完成
        self.progress_bar.setValue(len(windows))
        self.progress_bar.setFormat("分析完成!")
        self.btn_analyze.setEnabled(True)

        self.status_label.setText(f"分帧分析完成！共处理 {len(windows)} 个滑动窗口。")
        self.log("所有滑动窗口呼吸提取完毕！已生成联动视图。")

        self.combo_window_select.setCurrentIndex(0)
        self.plot_sliding_window_results()

    # ================= 算法调度封装 =================
    def _call_algorithm(self, method, frames, fs, progress_callback=None):
        """统一调度所有算法"""
        if method == "EMD":
            return algorithms.extract_emd(frames, fs)
        elif method == "VMD":
            return algorithms.extract_vmd(frames, fs)
        elif method == "AFD":
            return algorithms.extract_afd(frames, fs)
        elif method == "VMD_FPR":
            return algorithms.extract_vmd_fpr(frames, fs)
        elif method == "SMVMD":
            return algorithms.extract_smvmd(frames, fs)
        elif method == "MVMD":
            return algorithms.extract_mvmd(frames, fs)
        elif method == "Multi-ROI ICA":
            return algorithms.extract_multi_roi_ica(frames, fs)
        else:
            raise ValueError(f"未知算法: {method}")

    def _calc_bpm(self, signal, fs):
        """统一 BPM 计算"""
        if self.bpm_method_selector.currentText() == "FPR (特征点法)":
            return base.calculate_bpm_fpr(signal, fs)
        else:
            return base.calculate_bpm(signal, fs)

    # ================= 窗口切换 =================
    def on_window_selection_changed(self, idx):
        if not self.cb_sliding_window.isChecked() or idx < 0:
            return
        self.plot_sliding_window_results()

    # ================= 绘图：空间均值 =================
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
        ax.set_title(f"{title} (阈值 > {threshold})", color='#e4e4e7',
                      fontsize=14, fontweight='bold')
        ax.set_xlabel("时间 (秒)", color='#e4e4e7')
        ax.set_ylabel("受力点平均 ADC 强度", color='#e4e4e7')
        ax.tick_params(colors='#a1a1aa')
        ax.grid(True, color='#3f3f46', linestyle='--', alpha=0.5)

        for spine in ax.spines.values():
            spine.set_color('#3f3f46')

        self.canvas.draw()

    # ================= 绘图：全局单结果 =================
    def plot_single_result(self, signal, bpm, method):
        self.figure.clear()
        self.figure.patch.set_facecolor('#1a1a1e')

        # 上半部：呼吸波形
        ax_wave = self.figure.add_subplot(211)
        ax_wave.set_facecolor('#1e1e24')

        time_axis = np.arange(len(signal)) / self.fs
        ax_wave.plot(time_axis, signal, color='#ef4444', linewidth=2, label='提取的呼吸分量')
        ax_wave.set_title(f"全局呼吸波形 ({method}) | 呼吸率: {bpm:.1f} BPM",
                          color='#e4e4e7', fontsize=12, fontweight='bold')
        ax_wave.set_xlabel("时间 (秒)", color='#e4e4e7')
        ax_wave.set_ylabel("归一化强度", color='#e4e4e7')
        ax_wave.tick_params(colors='#a1a1aa')
        ax_wave.grid(True, color='#3f3f46', linestyle='--', alpha=0.5)
        ax_wave.legend(facecolor='#24242b', edgecolor='#3f3f46', labelcolor='#e4e4e7')
        for spine in ax_wave.spines.values():
            spine.set_color('#3f3f46')

        # 下半部：PSD
        ax_psd = self.figure.add_subplot(212)
        ax_psd.set_facecolor('#1e1e24')

        from scipy.signal import welch
        nperseg = min(len(signal), 256)
        f, psd = welch(signal, self.fs, nperseg=nperseg)

        ax_psd.plot(f, psd, color='#f59e0b', linewidth=2, label='功率谱密度 (PSD)')
        ax_psd.axvspan(0.1, 0.5, color='green', alpha=0.1, label="呼吸带 (0.1-0.5 Hz)")
        ax_psd.set_xlim(0, 1.5)
        ax_psd.set_title("全局呼吸信号 PSD 响应", color='#e4e4e7',
                          fontsize=12, fontweight='bold')
        ax_psd.set_xlabel("频率 (Hz)", color='#e4e4e7')
        ax_psd.set_ylabel("功率谱密度", color='#e4e4e7')
        ax_psd.tick_params(colors='#a1a1aa')
        ax_psd.grid(True, color='#3f3f46', linestyle='--', alpha=0.5)
        ax_psd.legend(facecolor='#24242b', edgecolor='#3f3f46', labelcolor='#e4e4e7')

        self._mark_psd_peak(ax_psd, f, psd, (0.1, 0.5), label="呼吸率")

        for spine in ax_psd.spines.values():
            spine.set_color('#3f3f46')

        self.figure.tight_layout()
        self.canvas.draw()

    # ================= 绘图：滑窗三子图 =================
    def plot_sliding_window_results(self):
        if not self.sliding_results:
            return

        self.figure.clear()
        self.figure.patch.set_facecolor('#1a1a1e')

        current_idx = self.combo_window_select.currentIndex()
        if current_idx < 0:
            current_idx = 0

        num_windows = len(self.sliding_results)
        is_merged_view = (current_idx == num_windows)

        # ---- 子图1: BPM 趋势 ----
        ax_trend = self.figure.add_subplot(311)
        ax_trend.set_facecolor('#1e1e24')

        times_sec = [r['center_time'] for r in self.sliding_results]
        bpms = [r['bpm'] for r in self.sliding_results]

        ax_trend.plot(times_sec, bpms, color='#3b82f6', marker='o',
                       markersize=6, linewidth=2, label='BPM 趋势')

        if not is_merged_view:
            sel_time = times_sec[current_idx]
            sel_bpm = bpms[current_idx]
            ax_trend.scatter(sel_time, sel_bpm, color='#ef4444', s=120,
                              zorder=5, label='当前选中窗口')
            ax_trend.axvline(x=sel_time, color='#ef4444', linestyle='--', alpha=0.7)
        else:
            ax_trend.axhspan(min(bpms) - 1, max(bpms) + 1, color='#ef4444',
                              alpha=0.08, label='全局融合段')

        ax_trend.set_title("呼吸率 (BPM) 变化趋势", color='#e4e4e7',
                            fontsize=11, fontweight='bold')
        ax_trend.set_xlabel("时间 (秒)", color='#e4e4e7')
        ax_trend.set_ylabel("BPM", color='#e4e4e7')
        ax_trend.tick_params(colors='#a1a1aa')
        ax_trend.grid(True, color='#3f3f46', linestyle='--', alpha=0.5)
        ax_trend.legend(facecolor='#24242b', edgecolor='#3f3f46', labelcolor='#e4e4e7')
        for spine in ax_trend.spines.values():
            spine.set_color('#3f3f46')

        # ---- 子图2: 呼吸波形 ----
        ax_wave = self.figure.add_subplot(312)
        ax_wave.set_facecolor('#1e1e24')

        if is_merged_view:
            # 加窗重叠相加融合拼接
            total_frames = len(self.clean_frames)
            wave = np.zeros(total_frames)
            weights = np.zeros(total_frames)

            for r in self.sliding_results:
                start_f = int(r['start_time'] * self.fs)
                end_f = int(r['end_time'] * self.fs)
                L = min(len(r['wave']), end_f - start_f)
                if L <= 0:
                    continue

                # Tukey-like 渐变窗
                w = np.ones(L)
                taper_ratio = 0.15
                taper_len = int(L * taper_ratio)
                if taper_len > 1:
                    taper = 0.05 + 0.95 * (0.5 * (1.0 - np.cos(
                        np.pi * np.arange(taper_len) / (taper_len - 1))))
                    w[:taper_len] = taper
                    w[-taper_len:] = taper[::-1]

                wave[start_f:start_f + L] += r['wave'][:L] * w
                weights[start_f:start_f + L] += w

            valid_mask = weights > 1e-5
            wave[valid_mask] /= weights[valid_mask]

            time_axis = np.arange(total_frames) / self.fs
            ax_wave.plot(time_axis, wave, color='#ec4899', linewidth=1.5,
                          label='滑窗加窗融合拼接波形')
            ax_wave.set_title("完整呼吸波形 (全段滑窗平滑融合)", color='#e4e4e7',
                              fontsize=11, fontweight='bold')
            ax_wave.set_xlabel("时间 (秒)", color='#e4e4e7')
        else:
            selected_res = self.sliding_results[current_idx]
            wave = selected_res['wave']
            start_time = selected_res['start_time']
            end_time = selected_res['end_time']
            bpm = selected_res['bpm']

            if self.cb_align_time.isChecked():
                time_axis = np.linspace(0, end_time - start_time, len(wave))
                ax_wave.plot(time_axis, wave, color='#10b981', linewidth=2,
                              label='提取的呼吸分量')
                ax_wave.set_title(
                    f"分帧明细 (对齐): 0.0s - {end_time - start_time:.1f}s | "
                    f"BPM: {bpm:.1f} (实际: {start_time:.1f}s - {end_time:.1f}s)",
                    color='#e4e4e7', fontsize=11, fontweight='bold')
                ax_wave.set_xlabel("相对时间 (秒)", color='#e4e4e7')
            else:
                time_axis = np.linspace(start_time, end_time, len(wave))
                ax_wave.plot(time_axis, wave, color='#10b981', linewidth=2,
                              label='提取的呼吸分量')
                ax_wave.set_title(
                    f"分帧明细: {start_time:.1f}s - {end_time:.1f}s | "
                    f"BPM: {bpm:.1f}",
                    color='#e4e4e7', fontsize=11, fontweight='bold')
                ax_wave.set_xlabel("绝对时间 (秒)", color='#e4e4e7')

        ax_wave.set_ylabel("幅值 (归一化)", color='#e4e4e7')
        ax_wave.tick_params(colors='#a1a1aa')
        ax_wave.grid(True, color='#3f3f46', linestyle='--', alpha=0.5)
        ax_wave.legend(facecolor='#24242b', edgecolor='#3f3f46', labelcolor='#e4e4e7')
        for spine in ax_wave.spines.values():
            spine.set_color('#3f3f46')

        # ---- 子图3: PSD ----
        ax_psd = self.figure.add_subplot(313)
        ax_psd.set_facecolor('#1e1e24')

        from scipy.signal import welch
        nperseg = min(len(wave), 256)
        f, psd = welch(wave, self.fs, nperseg=nperseg)

        ax_psd.plot(f, psd, color='#f59e0b', linewidth=2, label='PSD')
        ax_psd.axvspan(0.1, 0.5, color='green', alpha=0.1, label="呼吸带 (0.1-0.5 Hz)")
        ax_psd.set_xlim(0, 1.5)

        if is_merged_view:
            ax_psd.set_title("融合波形全局 PSD 响应", color='#e4e4e7',
                              fontsize=11, fontweight='bold')
        else:
            ax_psd.set_title(f"当前分帧 PSD | 呼吸率: {bpm:.1f} BPM",
                              color='#e4e4e7', fontsize=11, fontweight='bold')

        ax_psd.set_xlabel("频率 (Hz)", color='#e4e4e7')
        ax_psd.set_ylabel("功率谱密度", color='#e4e4e7')
        ax_psd.tick_params(colors='#a1a1aa')
        ax_psd.grid(True, color='#3f3f46', linestyle='--', alpha=0.5)
        ax_psd.legend(facecolor='#24242b', edgecolor='#3f3f46', labelcolor='#e4e4e7')

        self._mark_psd_peak(ax_psd, f, psd, (0.1, 0.5), label="呼吸率")

        for spine in ax_psd.spines.values():
            spine.set_color('#3f3f46')

        self.figure.tight_layout()
        self.canvas.draw()

    # ================= PSD 谱峰标注 =================
    def _mark_psd_peak(self, ax, freq, psd, band, label="主峰"):
        """在 PSD 子图上标注指定频段内的最强谱峰"""
        valid_mask = (freq >= band[0]) & (freq <= band[1])
        if np.any(valid_mask):
            f_roi = freq[valid_mask]
            psd_roi = psd[valid_mask]
            peak_idx = np.argmax(psd_roi)
            peak_freq = f_roi[peak_idx]
            peak_val = psd_roi[peak_idx]
            peak_bpm = peak_freq * 60.0

            ax.scatter(peak_freq, peak_val, color='red', s=45, zorder=5)
            ax.axvline(x=peak_freq, color='red', linestyle='--', alpha=0.5)
            ax.text(peak_freq + 0.02, peak_val * 0.8,
                    f"{label}: {peak_bpm:.1f} BPM\n({peak_freq:.3f} Hz)",
                    color='red', fontweight='bold', fontsize=9)

    # ================= 键盘导航 =================
    def keyPressEvent(self, event):
        """左右方向键切换滑窗"""
        if not self.combo_window_select.isEnabled():
            super().keyPressEvent(event)
            return

        # 输入框有焦点时不拦截
        focused_widget = QApplication.focusWidget()
        if isinstance(focused_widget, QLineEdit):
            super().keyPressEvent(event)
            return

        current_idx = self.combo_window_select.currentIndex()
        count = self.combo_window_select.count()

        if event.key() == Qt.Key_Left:
            if current_idx > 0:
                self.combo_window_select.setCurrentIndex(current_idx - 1)
                self.log(f"← 切换至上一窗口: {self.combo_window_select.currentText()}")
        elif event.key() == Qt.Key_Right:
            if current_idx < count - 1:
                self.combo_window_select.setCurrentIndex(current_idx + 1)
                self.log(f"→ 切换至下一窗口: {self.combo_window_select.currentText()}")
        else:
            super().keyPressEvent(event)

    # ================= 清理 =================
    def closeEvent(self, event):
        sys.stdout = self.stdout_bak
        super().closeEvent(event)


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

    app = QApplication(sys.argv)
    window = BreathSlidingWindowUI()
    window.show()
    sys.exit(app.exec_())
