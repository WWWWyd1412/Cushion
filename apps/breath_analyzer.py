#!/usr/bin/env python3
"""
Breath Analyzer — 呼吸信号滑窗分析 GUI
========================================
基于 cushion 包的新架构入口。

支持算法: EMD / VMD / AFD / VMD_FPR(MAPE) / GOA-VMD / SMVMD / MVMD / Multi-ROI ICA
"""

import sys
import os
import numpy as np

# 确保项目根目录在 sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# conda 环境 PyQt5 DLL 修复
_conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)
_qt_dll_path = os.path.join(_conda_prefix, 'Library', 'bin')
if os.path.isdir(_qt_dll_path) and _qt_dll_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _qt_dll_path + os.pathsep + os.environ.get('PATH', '')

import matplotlib
matplotlib.use('Qt5Agg')

# --- 中文字体配置 (解决 matplotlib 中文乱码) ---
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QMessageBox, QComboBox, QGroupBox, QLineEdit,
                             QCheckBox, QTextEdit, QProgressBar)
from PyQt5.QtCore import Qt

# --- cushion 包导入 ---
from cushion.core import load_pressure_txt, get_session_info, Preprocessor, clean_dataset
from cushion.core.signal_utils import smooth_signal, calculate_bpm_peak, calculate_bpm_fpr

# 算法导入
from cushion.algorithms.decomposition.emd import extract_emd
from cushion.algorithms.decomposition.vmd import extract_vmd
from cushion.algorithms.decomposition.smvmd import extract_smvmd
from cushion.algorithms.decomposition.mvmd import extract_mvmd
from cushion.algorithms.fusion.ica import extract_multi_roi_ica

from cushion.breath.config import BreathConfig as CFG
from cushion.ui.theme import DARK_THEME_QSS
from cushion.ui.widgets import TextRedirector
from cushion.ui.sliding_window import (generate_windows, overlap_add_fusion,
                                        compute_bpm_statistics)


# ============================================================================
# 算法分发映射
# ============================================================================
ALGO_REGISTRY = {
    "EMD":             lambda frames, fs: extract_emd(frames, fs, **CFG.to_dict()),
    "VMD":             lambda frames, fs: extract_vmd(frames, fs, K=CFG.VMD_K, alpha=CFG.VMD_ALPHA, **{k: v for k, v in CFG.to_dict().items() if k in ('freq_band', 'wavelet_alpha')}),
    "VMD_FPR":         lambda frames, fs: _extract_vmd_mape(frames, fs),
    "SMVMD":           lambda frames, fs: extract_smvmd(frames, fs),
    "MVMD":            lambda frames, fs: extract_mvmd(frames, fs),
    "Multi-ROI ICA":   lambda frames, fs: extract_multi_roi_ica(frames, fs),
}


def _extract_vmd_mape(frames, fs):
    """VMD-MAPE: K 值自适应寻优"""
    from cushion.core.signal_utils import butter_bandpass_filter, wavelet_denoise
    from cushion.algorithms.base import get_dual_roi_mean, reconstruct_multicomponent_with_snr
    from vmdpy import VMD

    signal_1d = get_dual_roi_mean(frames, fs=fs, freq_band=CFG.FREQ_BAND, wavelet_alpha=0.5)
    if len(signal_1d) < 100:
        return signal_1d

    mapes = []
    best_u = None
    for k in range(2, 11):
        u, _, _ = VMD(signal_1d, alpha=2000, tau=0, K=k, DC=0, init=1, tol=1e-7)
        res = signal_1d - np.sum(u, axis=0)
        mape = np.sum(res ** 2) / np.sum(signal_1d ** 2)
        if len(mapes) > 0 and mape > mapes[-1]:
            break
        mapes.append(mape)
        best_u = u

    return reconstruct_multicomponent_with_snr(best_u, fs, freq_band=CFG.FREQ_BAND)


# ============================================================================
# AFD 简化实现
# ============================================================================
def _extract_afd(frames, fs):
    from scipy.signal import hilbert
    from cushion.algorithms.base import get_dual_roi_mean, reconstruct_multicomponent_with_snr

    signal_1d = get_dual_roi_mean(frames, fs=fs, freq_band=CFG.FREQ_BAND, wavelet_alpha=0.5)
    z = hilbert(signal_1d - np.mean(signal_1d))
    t = np.arange(len(z)) / fs

    residual = z.copy()
    components = []
    search_freqs = np.linspace(0.1, 0.5, 50)

    for _ in range(5):
        best_comp = None
        max_proj = -1
        for f in search_freqs:
            kernel = np.exp(1j * 2 * np.pi * f * t)
            proj = np.abs(np.vdot(residual, kernel))
            if proj > max_proj:
                max_proj = proj
                best_comp = (np.vdot(residual, kernel) / np.vdot(kernel, kernel)) * kernel
        if best_comp is not None:
            components.append(np.real(best_comp))
            residual -= best_comp

    return reconstruct_multicomponent_with_snr(np.array(components), fs, freq_band=CFG.FREQ_BAND)


ALGO_REGISTRY["AFD"] = _extract_afd


# ============================================================================
# 呼吸分析主窗口
# ============================================================================
class BreathAnalyzerWindow(QMainWindow):
    """呼吸信号滑窗分析主界面"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("呼吸信号滑窗分析系统 (EMD/VMD/AFD/VMD_FPR/SMVMD/MVMD/Multi-ROI ICA)")
        self.resize(1500, 950)

        self.file_path = None
        self.raw_times = None
        self.raw_frames = None
        self.clean_times = None
        self.clean_frames = None
        self.fs = 10.0
        self.sliding_results = []
        self.stdout_bak = sys.stdout

        self.setup_ui()
        sys.stdout = TextRedirector(self.log_from_stdout)

    def setup_ui(self):
        self.setStyleSheet(DARK_THEME_QSS)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ---- 左侧控制面板 ----
        control_panel = QVBoxLayout()
        control_panel.setSpacing(15)

        # 1. 数据加载
        group_load = QGroupBox("1. 数据加载与清洗")
        layout_load = QVBoxLayout(group_load)
        self.btn_select_file = QPushButton("选择压力 TXT 文件")
        self.btn_preprocess = QPushButton("一键加载与清洗数据")
        self.btn_preprocess.setEnabled(False)
        layout_load.addWidget(self.btn_select_file)
        layout_load.addWidget(self.btn_preprocess)
        control_panel.addWidget(group_load)

        # 2. 算法配置
        group_algo = QGroupBox("2. 算法与参数配置")
        layout_algo = QVBoxLayout(group_algo)
        self.algo_selector = QComboBox()
        self.algo_selector.addItems(list(ALGO_REGISTRY.keys()))
        self.bpm_selector = QComboBox()
        self.bpm_selector.addItems(["Peak (常规波峰法)", "FPR (特征点法)"])
        layout_algo.addWidget(QLabel("呼吸提取算法:"))
        layout_algo.addWidget(self.algo_selector)
        layout_algo.addWidget(QLabel("BPM 测量方法:"))
        layout_algo.addWidget(self.bpm_selector)
        control_panel.addWidget(group_algo)

        # 3. 滑窗配置
        group_slide = QGroupBox("3. 滑动窗口设置")
        layout_slide = QVBoxLayout(group_slide)
        self.cb_sliding = QCheckBox("启用滑动窗口分析")
        self.cb_sliding.setChecked(True)

        layout_w = QHBoxLayout()
        layout_w.addWidget(QLabel("窗口 (帧):"))
        self.edit_window = QLineEdit("250")
        layout_w.addWidget(self.edit_window)
        layout_w.addWidget(QLabel("步长 (帧):"))
        self.edit_step = QLineEdit("50")
        layout_w.addWidget(self.edit_step)
        layout_slide.addWidget(self.cb_sliding)
        layout_slide.addLayout(layout_w)
        control_panel.addWidget(group_slide)

        # 4. 执行
        group_run = QGroupBox("4. 执行分析")
        layout_run = QVBoxLayout(group_run)
        self.btn_run = QPushButton("▶ 开始滑窗分析")
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet("QPushButton { background-color: #10b981; }")
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout_run.addWidget(self.btn_run)
        layout_run.addWidget(self.progress)
        control_panel.addWidget(group_run)

        # 5. 导航
        group_nav = QGroupBox("5. 窗口导航")
        layout_nav = QVBoxLayout(group_nav)
        self.window_selector = QComboBox()
        self.window_selector.setEnabled(False)
        layout_nav.addWidget(QLabel("选择分析窗口:"))
        layout_nav.addWidget(self.window_selector)
        control_panel.addWidget(group_nav)

        # 6. 日志面板
        group_log = QGroupBox("6. 运行日志")
        layout_log = QVBoxLayout(group_log)
        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setMaximumHeight(180)
        layout_log.addWidget(self.log_panel)
        control_panel.addWidget(group_log)

        control_panel.addStretch()

        # ---- 右侧绘图区 ----
        right_layout = QVBoxLayout()
        self.figure = plt.figure(figsize=(12, 9))
        self.figure.patch.set_facecolor('#1a1a1e')
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        right_layout.addWidget(NavigationToolbar(self.canvas, self))

        main_layout.addLayout(control_panel, 3)
        main_layout.addLayout(right_layout, 7)

        # ---- 信号连接 ----
        self.btn_select_file.clicked.connect(self.select_file)
        self.btn_preprocess.clicked.connect(self.load_and_clean)
        self.btn_run.clicked.connect(self.run_analysis)
        self.window_selector.currentIndexChanged.connect(self.plot_window)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择压力矩阵 TXT 文件", "../data",
            "Text Files (*.txt);;All Files (*)")
        if path:
            self.file_path = path
            self.btn_preprocess.setEnabled(True)
            self.log(f"已选择文件: {path}")

    def load_and_clean(self):
        self.log("正在加载数据...")
        self.raw_times, self.raw_frames = load_pressure_txt(self.file_path)
        if self.raw_frames is None:
            QMessageBox.critical(self, "错误", "数据加载失败!")
            return

        get_session_info(self.raw_times, self.raw_frames)

        self.log("正在清洗数据...")
        self.clean_times, self.clean_frames = clean_dataset(
            self.raw_times, self.raw_frames, calib_count=10, fs=self.fs,
            trim_seconds=20, use_gaussian=True, gaussian_sigma=CFG.GAUSSIAN_SIGMA)

        self.btn_run.setEnabled(True)
        self.log(f"清洗完成: {len(self.clean_frames)} 帧有效数据")

    def run_analysis(self):
        algo_name = self.algo_selector.currentText()
        use_sliding = self.cb_sliding.isChecked()
        use_fpr = "FPR" in self.bpm_selector.currentText()

        self.progress.setVisible(True)
        self.sliding_results = []
        self.log(f"算法: {algo_name} | BPM方法: {'FPR' if use_fpr else 'Peak'}")

        extract_fn = ALGO_REGISTRY.get(algo_name)
        if extract_fn is None:
            QMessageBox.warning(self, "警告", f"未知算法: {algo_name}")
            return

        if use_sliding:
            window_size = int(self.edit_window.text())
            step_size = int(self.edit_step.text())
            windows = generate_windows(len(self.clean_frames), window_size, step_size)
            self.log(f"滑窗分析: {len(windows)} 个窗口, 窗口={window_size}帧, 步长={step_size}帧")

            waveforms = []
            bpm_list = []
            for idx, (start, end) in enumerate(windows):
                win_frames = self.clean_frames[start:end]
                wave = extract_fn(win_frames, self.fs)
                wave = smooth_signal(wave, window_size=CFG.SAVGOL_WINDOW,
                                     polyorder=CFG.SAVGOL_ORDER)
                waveforms.append(wave)

                if use_fpr:
                    bpm = calculate_bpm_fpr(wave, self.fs, min_dist_sec=CFG.BPM_MIN_DIST_SEC)
                else:
                    bpm = calculate_bpm_peak(wave, self.fs, min_dist_sec=CFG.BPM_MIN_DIST_SEC)
                bpm_list.append(bpm)

                self.sliding_results.append({
                    'start': start, 'end': end, 'waveform': wave, 'bpm': bpm
                })

                pct = int((idx + 1) / len(windows) * 100)
                self.progress.setValue(pct)

            stats = compute_bpm_statistics(bpm_list)
            self.log(f"BPM 统计: 均值={stats['mean']:.1f}, 标准差={stats['std']:.1f}, "
                      f"范围=[{stats['min']:.1f}, {stats['max']:.1f}], "
                      f"有效窗口={stats['count']}/{len(bpm_list)}")
        else:
            # 全段分析
            wave = extract_fn(self.clean_frames, self.fs)
            wave = smooth_signal(wave, window_size=CFG.SAVGOL_WINDOW,
                                 polyorder=CFG.SAVGOL_ORDER)
            if use_fpr:
                bpm = calculate_bpm_fpr(wave, self.fs)
            else:
                bpm = calculate_bpm_peak(wave, self.fs)
            self.sliding_results = [{'start': 0, 'end': len(self.clean_frames),
                                      'waveform': wave, 'bpm': bpm}]
            self.log(f"全段分析 BPM: {bpm:.1f}")

        self.window_selector.clear()
        self.window_selector.addItems([f"窗口 {i+1} (帧 {r['start']}-{r['end']})"
                                        for i, r in enumerate(self.sliding_results)])
        self.window_selector.setEnabled(True)
        self.progress.setVisible(False)

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

    def plot_window(self):
        idx = self.window_selector.currentIndex()
        if idx < 0 or idx >= len(self.sliding_results):
            return

        self.figure.clear()
        self.figure.patch.set_facecolor('#1a1a1e')

        r = self.sliding_results[idx]

        # ---- 子图1: BPM 趋势 ----
        ax_trend = self.figure.add_subplot(311)
        ax_trend.set_facecolor('#1e1e24')

        bpm_vals = [s['bpm'] for s in self.sliding_results]
        times_sec = [s['start'] / self.fs for s in self.sliding_results]
        ax_trend.plot(times_sec, bpm_vals, color='#3b82f6', marker='o',
                       markersize=6, linewidth=2, label='BPM 趋势')

        # 高亮当前窗口
        sel_time = times_sec[idx]
        sel_bpm = bpm_vals[idx]
        ax_trend.scatter(sel_time, sel_bpm, color='#ef4444', s=120, zorder=5,
                          label='当前选中窗口')
        ax_trend.axvline(x=sel_time, color='#ef4444', linestyle='--', alpha=0.7)

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

        wave = r['waveform']
        start_time = r['start'] / self.fs
        end_time = r['end'] / self.fs
        bpm = r['bpm']
        time_axis = np.linspace(start_time, end_time, len(wave))
        ax_wave.plot(time_axis, wave, color='#10b981', linewidth=2,
                      label='提取的呼吸分量')
        ax_wave.set_title(
            f"分帧明细: {start_time:.1f}s - {end_time:.1f}s | BPM: {bpm:.1f}",
            color='#e4e4e7', fontsize=11, fontweight='bold')
        ax_wave.set_xlabel("时间 (秒)", color='#e4e4e7')
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

        ax_psd.plot(f, psd, color='#f59e0b', linewidth=2, label='功率谱密度 (PSD)')
        ax_psd.axvspan(0.1, 0.5, color='green', alpha=0.1, label="呼吸带 (0.1-0.5 Hz)")
        ax_psd.set_xlim(0, 1.5)
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

    def log(self, msg):
        self.log_panel.append(msg)

    def log_from_stdout(self, text):
        self.log_panel.append(text)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right and self.window_selector.isEnabled():
            idx = self.window_selector.currentIndex()
            if idx < self.window_selector.count() - 1:
                self.window_selector.setCurrentIndex(idx + 1)
        elif event.key() == Qt.Key_Left and self.window_selector.isEnabled():
            idx = self.window_selector.currentIndex()
            if idx > 0:
                self.window_selector.setCurrentIndex(idx - 1)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        sys.stdout = self.stdout_bak
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = BreathAnalyzerWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
