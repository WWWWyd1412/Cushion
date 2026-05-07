import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QMessageBox, QComboBox, QGroupBox)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# 导入自定义模块
import data_loader
import preprocess
import algorithms  # 确保 algorithms 文件夹下有 __init__.py 并已正确配置


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("压力矩阵分析系统")
        self.resize(1400, 900)

        # 数据存储变量
        self.file_path = None
        self.raw_times = None
        self.raw_frames = None
        self.clean_times = None
        self.clean_frames = None
        self.breath_signal = None

        self.fs = 10.0  # 默认采样率，后续可通过时间戳动态计算

        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- 左侧控制面板 ---
        control_panel = QVBoxLayout()
        control_panel.setSpacing(15)

        # 第一步：文件选择
        group1 = QGroupBox("1. 数据导入")
        layout1 = QVBoxLayout(group1)
        self.btn_select = QPushButton("选择 TXT 文件")
        self.btn_load = QPushButton("加载原始数据")
        self.btn_load.setEnabled(False)
        layout1.addWidget(self.btn_select)
        layout1.addWidget(self.btn_load)
        control_panel.addWidget(group1)

        # 第二步：预处理
        group2 = QGroupBox("2. 数据清洗")
        layout2 = QVBoxLayout(group2)
        self.btn_preprocess = QPushButton("执行预处理(剔除坏帧)")
        self.btn_preprocess.setEnabled(False)
        layout2.addWidget(self.btn_preprocess)
        control_panel.addWidget(group2)

        # 第三步：算法提取
        group3 = QGroupBox("3. 呼吸提取算法")
        layout3 = QVBoxLayout(group3)
        self.algo_selector = QComboBox()
        self.algo_selector.addItems(["EMD", "VMD", "AFD", "VMD_FPR"])
        self.btn_analyze = QPushButton("提取呼吸波形")
        self.btn_analyze.setEnabled(False)
        layout3.addWidget(QLabel("选择方法:"))
        layout3.addWidget(self.algo_selector)
        layout3.addWidget(self.btn_analyze)
        control_panel.addWidget(group3)

        # 状态显示
        self.status_label = QLabel("状态: 请选择文件")
        self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        self.status_label.setWordWrap(True)
        control_panel.addWidget(self.status_label)
        control_panel.addStretch()

        # 绑定槽函数
        self.btn_select.clicked.connect(self.step1_select_file)
        self.btn_load.clicked.connect(self.step2_load_data)
        self.btn_preprocess.clicked.connect(self.step3_preprocess)
        self.btn_analyze.clicked.connect(self.step4_analyze)

        main_layout.addLayout(control_panel, 1)

        # --- 右侧绘图区 ---
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)

        self.figure = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)

        # 添加交互工具栏（关键步骤）
        self.toolbar = NavigationToolbar(self.canvas, self)

        plot_layout.addWidget(self.toolbar)  # 将工具栏添加到布局
        plot_layout.addWidget(self.canvas)

        main_layout.addWidget(plot_container, 4)  # 将容器添加到主布局[cite: 10]

    # --- 逻辑处理步骤 ---

    def step1_select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择压力数据", "", "Text Files (*.txt)")
        if path:
            self.file_path = path
            self.status_label.setText(f"已选择文件: {path}")
            self.btn_load.setEnabled(True)

    def step2_load_data(self):
        try:
            # 调用加载器解析数据
            self.raw_times, self.raw_frames = data_loader.load_pressure_txt(self.file_path)
            self.status_label.setText(f"加载成功: {len(self.raw_frames)} 帧数据")

            # 由于16行/列是坏点，绘图时显示空间平均趋势[cite: 11]
            self.plot_spatial_mean(self.raw_frames, "原始数据空间平均趋势 (含异常尖峰)")
            self.btn_preprocess.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败: {e}")

    def step3_preprocess(self):
        if self.raw_frames is None: return

        # 调用预处理剔除坏帧（如 21677）
        self.clean_times, self.clean_frames = preprocess.clean_dataset(self.raw_times, self.raw_frames)
        self.status_label.setText(f"预处理完成！有效帧: {len(self.clean_frames)} (剔除坏帧)")

        # 显示清洗后的趋势
        self.plot_spatial_mean(self.clean_frames, "清洗后空间平均趋势 (已剔除坏帧)")
        self.btn_analyze.setEnabled(True)

    def step4_analyze(self):
        if self.clean_frames is None: return

        method = self.algo_selector.currentText()
        self.status_label.setText(f"正在使用 {method} 算法提取...")
        QApplication.processEvents()

        try:
            # 1. 执行算法提取
            if method == "EMD":
                raw_breath = algorithms.extract_emd(self.clean_frames, self.fs)
            elif method == "VMD":
                raw_breath = algorithms.extract_vmd(self.clean_frames, self.fs)
            elif method == "VMD_FPR":
                raw_breath = algorithms.extract_vmd_fpr(self.clean_frames, self.fs)
            else:
                raw_breath = algorithms.extract_afd(self.clean_frames, self.fs)

            # 2. 消除开头突发波峰：切除前 100 帧 (约 10 秒数据)
            # 因为图中明显的干扰集中在开头
            offset = 100
            if len(raw_breath) > offset:
                processed_signal = raw_breath[offset:]
            else:
                processed_signal = raw_breath

            # 3. 结果平滑滤波
            self.breath_signal = algorithms.smooth_respiration_signal(processed_signal)

            # 4. 重新计算 BPM (基于处理后的信号)
            if method == "VMD_FPR":
                bpm = algorithms.calculate_bpm_fpr(self.breath_signal, self.fs)
                method_title = "VMD-FPR 重构"
            else:
                bpm = algorithms.calculate_bpm(self.breath_signal, self.fs)
                method_title = method
            # 5. 绘图显示
            self.plot_final_result(self.breath_signal, f"提取的呼吸波形 ({method_title}) | 检测频率: {bpm:.1f} BPM")
            self.status_label.setText(f"分析完成！已自动平滑并切除起始干扰区，呼吸率: {bpm:.1f}")

        except Exception as e:
            QMessageBox.critical(self, "算法错误", f"计算失败: {e}")

    # --- 绘图辅助函数 ---

    def plot_spatial_mean(self, frames, title):
        """计算活跃区域均值，确保UI波形幅度与分析一致[cite: 24, 26]"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        avg_trend = []
        threshold = 100  # 你要求的筛选阈值

        for f in frames:
            # 动态筛选每一帧中真正受力的点
            active_points = f[f > threshold]
            if active_points.size > 0:
                avg_trend.append(np.mean(active_points))
            else:
                # 回退到非零点平均[cite: 24]
                non_zero = f[f > 35]
                avg_trend.append(np.mean(non_zero) if non_zero.size > 0 else 0)

        ax.plot(avg_trend, color='#3498db', linewidth=1)
        ax.set_title(f"{title} (Dynamic Threshold > {threshold})")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Active ADC Intensity")  # 此时坐标轴应在400以上
        ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def plot_final_result(self, signal, title):
        """绘制最终提取的一维呼吸信号"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ax.plot(signal, color='#e74c3c', linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Normalized Intensity")
        ax.grid(True, alpha=0.3)
        self.canvas.draw()


if __name__ == "__main__":
    # 配置中文支持
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())