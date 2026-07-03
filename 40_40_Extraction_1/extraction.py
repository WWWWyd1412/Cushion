# -*- coding: utf-8 -*-
"""
SCU 40x40 柔性压力阵列数据采集与可视化系统 v5.0
功能：
1. 串口配置与自动检测，支持串口通信波特率选择。
2. 动态调节大小端（Big Endian / Little Endian）解析，修复因字节序不对导致的数值混乱。
3. 零点校准功能，扣除背景底噪。
4. 空间滤波器：中值滤波 + 高斯平滑，提升热力图视觉效果。
5. 实时热力图渲染，支持多种调色板。
6. 实时 40x40 原始数据网格矩阵（QTableWidget）展示（分频刷新，防UI卡顿）。
7. 数据自动保存，可自定义文件夹，格式为 txt，保存时间戳和扁平化的矩阵数据。
8. 仿真测试模式，模拟流动压力斑点。
9. 🆕 实时呼吸信号采集与波形显示（支持均值/VMD/EMD/AFD 算法）。
10. 🆕 实时心跳信号采集与波形显示（支持均值/VMD/EMD 算法）。
"""

import sys
import os
import time
import serial  # type: ignore
import serial.tools.list_ports  # type: ignore
import numpy as np
from datetime import datetime
from scipy.ndimage import median_filter, gaussian_filter

# 预加载 Qt _conda DLL，兼容 Python 3.7（os.environ['PATH'] 对当前进程无效）
import ctypes as _ctypes
_qt_lib = os.path.join(os.environ.get('CONDA_PREFIX', sys.prefix), 'Library', 'bin')
for _dll in ['Qt5Core_conda', 'Qt5Gui_conda', 'Qt5Widgets_conda', 'Qt5Network_conda']:
    try:
        _ctypes.CDLL(os.path.join(_qt_lib, _dll + '.dll'))
    except OSError:
        pass

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
                             QHeaderView, QPushButton, QComboBox, QCheckBox, QFileDialog,
                             QMessageBox, QGroupBox, QSplitter, QScrollArea, QTabWidget)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QRectF, QTimer
from PyQt5.QtGui import QFont, QColor
import pyqtgraph as pg

# 导入算法模块
from algorithms.base import smooth_signal, calculate_bpm, calculate_bpm_fpr, get_spatial_sum
from algorithms.breath_extract import (
    extract_breath_mean, extract_breath_vmd,
    extract_breath_emd, extract_breath_afd,
    extract_breath_vmd_mape, extract_breath_goa_vmd,
    extract_breath_smvmd, extract_breath_mvmd,
    extract_breath_multi_roi_ica, extract_breath_acmd
)
from algorithms.heartbeat_extract import (
    extract_heartbeat_mean, extract_heartbeat_vmd, extract_heartbeat_emd,
    extract_heartbeat_acmd, extract_heartbeat_vme
)

# Premium 暗色主题样式表
DARK_STYLE = """
* {
    font-family: "Segoe UI", "Microsoft YaHei", "Segoe UI Semibold", sans-serif;
    font-size: 14px;
}
QMainWindow {
    background-color: #0b0f19;
}
QScrollArea {
    border: none;
    background-color: #0b0f19;
}
QGroupBox {
    font-weight: bold;
    font-size: 15px;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 10px;
    margin-top: 20px;
    padding-top: 18px;
    padding-bottom: 12px;
    padding-left: 12px;
    padding-right: 12px;
    background-color: #111827;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 15px;
    top: -2px;
    padding: 0 8px;
    background-color: #111827;
    color: #38bdf8;
}
QLabel {
    color: #94a3b8;
    font-size: 14px;
    min-height: 22px;
}
QLabel#TitleLabel {
    color: #f8fafc;
    font-weight: bold;
    font-size: 20px;
    min-height: 32px;
    margin-bottom: 10px;
}
QCheckBox {
    color: #e2e8f0;
    font-size: 14px;
    min-height: 28px;
}
QComboBox {
    background-color: #1f2937;
    color: #f1f5f9;
    border: 1px solid #4b5563;
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 32px;
}
QComboBox QAbstractItemView {
    background-color: #1f2937;
    color: #f1f5f9;
    selection-background-color: #2563eb;
}
QPushButton {
    background-color: #2563eb;
    color: #ffffff;
    font-weight: bold;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 14px;
    border: none;
    min-height: 36px;
}
QPushButton:hover {
    background-color: #3b82f6;
}
QPushButton:pressed {
    background-color: #1d4ed8;
}
QPushButton:disabled {
    background-color: #4b5563;
    color: #9ca3af;
}
QPushButton#StartBtn {
    background-color: #10b981;
}
QPushButton#StartBtn:hover {
    background-color: #34d399;
}
QPushButton#StopBtn {
    background-color: #ef4444;
}
QPushButton#StopBtn:hover {
    background-color: #f87171;
}
QPushButton#CalibBtn {
    background-color: #8b5cf6;
}
QPushButton#CalibBtn:hover {
    background-color: #a78bfa;
}
QTableWidget {
    background-color: #0f172a;
    gridline-color: #1e293b;
    border: 1px solid #1e293b;
    border-radius: 6px;
    color: #38bdf8;
    font-weight: bold;
}
QScrollBar:vertical {
    border: none;
    background: #0f172a;
    width: 8px;
}
QScrollBar::handle:vertical {
    background: #334155;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #475569;
}
QScrollBar:horizontal {
    border: none;
    background: #0f172a;
    height: 8px;
}
QScrollBar::handle:horizontal:hover {
    background: #475569;
}
QTabWidget::pane {
    border: 1px solid #1e293b;
    background-color: #0f172a;
    border-radius: 8px;
}
QTabBar::tab {
    background-color: #1e293b;
    color: #94a3b8;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 10px 20px;
    margin-right: 4px;
    font-size: 14px;
}
QTabBar::tab:selected {
    background-color: #0f172a;
    color: #f1f5f9;
    border-bottom: 2px solid #2563eb;
    font-weight: bold;
}
QTabBar::tab:hover {
    background-color: #334155;
    color: #f8fafc;
}
"""

# ──────────────────────────────────────────────
# 信号提取参数
# ──────────────────────────────────────────────
SIGNAL_BUFFER_SIZE = 168       # 滑动窗口长度：15秒 @ 11.2Hz
TARGET_SAMPLE_RATE = 11.2      # 目标信号采样率 (Hz)
ALGO_INTERVAL_MS = 1000        # 算法执行间隔 (ms)
BREATH_LOWCUT = 0.1            # 呼吸频段下限 (Hz)
BREATH_HIGHCUT = 0.5           # 呼吸频段上限 (Hz)
HEARTBEAT_LOWCUT = 0.8         # 心跳频段下限 (Hz)
HEARTBEAT_HIGHCUT = 2.2        # 心跳频段上限 (Hz)


class SerialThread(QThread):
    """
    后台串口数据采集线程
    """
    data_received = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.port = ''
        self.baudrate = 115200
        self.endianness = 'little'  # 'little' 或 'big'
        self.simulate = False
        self.running = False
        self.ser = None

    def run(self):
        if self.simulate:
            self.run_simulation()
            return

        try:
            # 打开串口
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.2)
            # 延时以保证硬件电平稳定
            time.sleep(0.5)
            self.ser.reset_input_buffer()

            # 部分微控制器可能需要发送 '1' 来启动数据流
            try:
                self.ser.write(b'1')
            except:
                pass

            self.running = True

            # 40x40 矩阵，双字节表示一个数据点，总共 3200 字节的数据负载。
            # 包头为 5A 01 95 6C (4字节) + 2字节 (如帧序号或长度等) + 3200字节 = 3206字节
            packet_len = 3206
            header_bytes = b'\x5A\x01\x95\x6C'
            buffer = bytearray()

            while self.running:
                if self.ser.in_waiting > 0:
                    buffer.extend(self.ser.read(self.ser.in_waiting))

                    while len(buffer) >= packet_len:
                        idx = buffer.find(header_bytes)
                        if idx == -1:
                            # 没找到包头，保留包头长度-1的尾部数据，防止包头被截断
                            buffer = buffer[-(len(header_bytes) - 1):]
                            break
                        if idx > 0:
                            # 丢弃包头之前的数据
                            buffer = buffer[idx:]
                            continue

                        # 缓冲区长度足够解析一整包
                        if len(buffer) >= packet_len:
                            raw_payload = buffer[6:packet_len]

                            # 选择字节序
                            dtype_str = '<u2' if self.endianness == 'little' else '>u2'
                            data = np.frombuffer(raw_payload, dtype=dtype_str).copy()

                            if data.size == 1600:
                                matrix = data.reshape((40, 40))
                                self.data_received.emit(matrix)  # type: ignore

                            # 移出当前已解析包
                            buffer = buffer[packet_len:]
                        else:
                            break
                else:
                    time.sleep(0.002)

            # 停止命令
            try:
                if self.ser is not None:
                    self.ser.write(b'2')
            except:
                pass
            if self.ser and self.ser.is_open:
                self.ser.close()
            self.ser = None

        except Exception as e:
            self.error_occurred.emit(str(e))  # type: ignore
            self.running = False

    def run_simulation(self):
        """
        生成逼真的仿真模拟数据，用来在无硬件连接时进行测试。
        模拟包含缓慢呼吸波动和较快心跳波动的压力信号。
        """
        self.running = True
        t = 0.0
        x = np.linspace(-3, 3, 40)
        y = np.linspace(-3, 3, 40)
        X, Y = np.meshgrid(x, y)

        while self.running:
            # 基础底噪 (ADC 12位精度通常有一定幅值的波动)
            base_noise = np.random.normal(240, 3, (40, 40)).astype(np.float64)

            # 移动的压力点斑（模拟坐姿压力分布）
            cx1, cy1 = 1.5 * np.sin(t * 0.05), 1.5 * np.cos(t * 0.07)
            cx2, cy2 = 1.2 * np.sin(t * 0.08 + np.pi / 4), 1.2 * np.cos(t * 0.04)

            blob1 = 1800 * np.exp(-((X - cx1)**2 + (Y - cy1)**2) / 0.8)
            blob2 = 1200 * np.exp(-((X - cx2)**2 + (Y - cy2)**2) / 0.5)

            # 叠加呼吸调制（缓慢的幅度变化，~0.25 Hz）
            breath_mod = 1.0 + 0.15 * np.sin(2 * np.pi * 0.25 * t / 33.0)

            # 叠加微弱心跳调制（较快的幅度变化，~1.2 Hz）
            heartbeat_mod = 1.0 + 0.03 * np.sin(2 * np.pi * 1.2 * t / 33.0)

            simulated_matrix = (base_noise + blob1 + blob2) * breath_mod * heartbeat_mod
            simulated_matrix = np.clip(simulated_matrix, 0, 4095).astype(np.uint16)

            self.data_received.emit(simulated_matrix)  # type: ignore
            time.sleep(0.03)  # 大约 33 FPS
            t += 1.0

    def stop(self):
        self.running = False


class RealTimeHeatmapView(pg.GraphicsLayoutWidget):
    """
    基于 PyQtGraph 的高性能压力热力图视图
    """
    mouse_moved_signal = pyqtSignal(int, int, float)

    def __init__(self):
        super().__init__()
        self.setBackground('#0f172a')
        self.view = self.addViewBox()
        self.view.setAspectLocked(True)  # 强制热力图为正方形，防止拉伸变形
        self.view.invertY(True)
        self.view.setContentsMargins(0, 0, 0, 0)
        self.view.setMouseEnabled(x=False, y=False)  # 禁用鼠标缩放与拖拽，防止影响视图
        self.view.setMenuEnabled(False)  # 禁用右键快捷菜单

        self.img = pg.ImageItem()
        self.img.setOpts(smooth=True)  # 默认开启双线性插值平滑渲染，消除噪点块
        self.view.addItem(self.img)
        self.view.setRange(QRectF(0, 0, 40, 40), disableAutoRange=True)

        # 默认使用 inferno 调色板
        self.colorbar = pg.ColorBarItem(values=(0, 2000), colorMap=pg.colormap.get('inferno'))
        self.colorbar.setImageItem(self.img)

        self.matrix = np.zeros((40, 40))
        scene = self.img.scene()
        if scene is not None:
            scene.sigMouseMoved.connect(self.on_mouse_moved)  # type: ignore

    def change_colormap(self, cmap_name):
        """
        切换色彩映射表
        """
        try:
            cmap = pg.colormap.get(cmap_name)
            self.colorbar.setColorMap(cmap)
        except Exception as e:
            print(f"Colormap change error: {e}")

    def draw_matrix(self, matrix):
        self.matrix = matrix

        current_min = np.min(matrix)
        current_max = np.max(matrix)
        if current_max <= current_min:
            current_max = current_min + 10

        self.img.setLevels([current_min, current_max])
        self.colorbar.setLevels((current_min, current_max))

        # 矩阵转置 (.T) 以匹配物理世界和数学坐标系的行和列
        self.img.setImage(matrix.T, autoLevels=False)

    def on_mouse_moved(self, pos):
        mouse_point = self.view.mapSceneToView(pos)
        c = int(mouse_point.x())
        r = int(mouse_point.y())
        if 0 <= r < 40 and 0 <= c < 40:
            val = self.matrix[r, c]
            self.mouse_moved_signal.emit(r, c, val)  # type: ignore


class MainWindow(QMainWindow):
    """
    主窗体 —— 40x40 压力阵列采集 + 实时呼吸/心跳信号提取
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle(" 40x40 柔性压力阵列采集")
        self.resize(1700, 920)
        self.setStyleSheet(DARK_STYLE)

        # ── 数据状态与处理变量 ──
        self.base_matrix = np.zeros((40, 40))
        self.is_calibrated = False
        self.calibrating = False

        # 存储路径默认在工作目录下的 data 文件夹
        self.save_path = os.path.join(os.getcwd(), "data")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.is_recording = False
        self.file_handle = None

        # 统计计数器
        self.frame_count = 0
        self.last_fps_time = time.time()

        # 保存上一帧供滤波器或UI共享
        self.current_processed_matrix = np.zeros((40, 40))
        # 3D 帧序列缓冲区 (用于高级时空算法，如 SMVMD, MVMD 等)
        self.frame_buffer = []

        # ── 🆕 信号提取变量 ──
        self.breath_buffer = np.zeros(SIGNAL_BUFFER_SIZE)
        self.heartbeat_buffer = np.zeros(SIGNAL_BUFFER_SIZE)
        self.breath_signal_smoothed = np.zeros(SIGNAL_BUFFER_SIZE)
        self.heartbeat_signal_filtered = np.zeros(SIGNAL_BUFFER_SIZE)
        self.breath_bpm = 0.0
        self.heartbeat_bpm = 0.0
        self.signal_frame_counter = 0
        self.downsample_factor = 3  # 约 33 FPS / 10 Hz ≈ 3，动态调整

        # 累计重叠区间计算相关的缓冲区与权重窗函数 (解决滑动窗口跳变问题)
        self.window_weights = np.sin(np.pi * np.linspace(0.1, 0.9, SIGNAL_BUFFER_SIZE))
        self.breath_algo_accum = np.zeros(SIGNAL_BUFFER_SIZE)
        self.breath_algo_weight = np.full(SIGNAL_BUFFER_SIZE, 1e-6)
        self.heartbeat_algo_accum = np.zeros(SIGNAL_BUFFER_SIZE)
        self.heartbeat_algo_weight = np.full(SIGNAL_BUFFER_SIZE, 1e-6)

        # 构建 UI
        self.setup_ui()

        # 串口线程初始化
        self.serial_thread = SerialThread()
        self.serial_thread.data_received.connect(self.on_data_received)  # type: ignore
        self.serial_thread.error_occurred.connect(self.on_serial_error)  # type: ignore

        # 初始化定时器用于限制 QTableWidget 网格的刷新频率（避免过高帧率引起UI卡顿）
        self.table_timer = QTimer()
        self.table_timer.timeout.connect(self.update_raw_table)
        self.table_timer.start(200)  # 5 Hz 刷新率

        # 🆕 算法执行定时器（每秒执行一次）
        self.algo_timer = QTimer()
        self.algo_timer.timeout.connect(self.run_algorithms)

    # ══════════════════════════════════════════════
    # UI 构建
    # ══════════════════════════════════════════════
    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 主水平布局
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # ── 左侧控制面板 (裹在独立的 QScrollArea 中，仅允许左侧设置栏内部滚动，绝不影响主界面) ──
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # type: ignore
        left_scroll.setMinimumWidth(390)
        left_scroll.setMaximumWidth(430)
        left_scroll.setStyleSheet("background-color: #0b0f19; border: none;")

        left_widget = QWidget()
        left_widget.setStyleSheet("background-color: #0b0f19;")
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 5, 0)
        left_layout.setSpacing(10)

        title_label = QLabel("📊 40x40 Matrix Controller")
        title_label.setObjectName("TitleLabel")
        left_layout.addWidget(title_label)

        # 1. 硬件连接组
        conn_group = QGroupBox("📡 硬件连接")
        conn_grid = QVBoxLayout(conn_group)
        conn_grid.setContentsMargins(10, 18, 10, 10)
        conn_grid.setSpacing(8)

        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("端口:"))
        self.port_combo = QComboBox()
        self.refresh_ports()
        port_layout.addWidget(self.port_combo, 2)

        self.refresh_ports_btn = QPushButton("🔄")
        self.refresh_ports_btn.setToolTip("刷新串口列表")
        self.refresh_ports_btn.setStyleSheet("padding: 4px 8px; font-size: 12px; background-color: #475569;")
        self.refresh_ports_btn.clicked.connect(self.refresh_ports)  # type: ignore
        port_layout.addWidget(self.refresh_ports_btn)
        conn_grid.addLayout(port_layout)

        baud_layout = QHBoxLayout()
        baud_layout.addWidget(QLabel("波特率:"))
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(['115200', '460800', '921600'])
        self.baud_combo.setCurrentText('115200')
        baud_layout.addWidget(self.baud_combo)
        conn_grid.addLayout(baud_layout)

        endian_layout = QHBoxLayout()
        endian_layout.addWidget(QLabel("字节序:"))
        self.endian_combo = QComboBox()
        self.endian_combo.addItems(['Little Endian (小端)', 'Big Endian (大端)'])
        self.endian_combo.setCurrentIndex(0)
        endian_layout.addWidget(self.endian_combo)
        conn_grid.addLayout(endian_layout)

        self.simulate_cb = QCheckBox("启用仿真模拟测试数据")
        conn_grid.addWidget(self.simulate_cb)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("▶ 开始采集")
        self.start_btn.setObjectName("StartBtn")
        self.start_btn.clicked.connect(self.start_acquisition)  # type: ignore
        btn_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■ 停止")
        self.stop_btn.setObjectName("StopBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_acquisition)  # type: ignore
        btn_row.addWidget(self.stop_btn)
        conn_grid.addLayout(btn_row)

        left_layout.addWidget(conn_group)

        # 2. 去噪与滤波设置组
        filter_group = QGroupBox("⚙️ 去噪与滤波")
        filter_layout = QVBoxLayout(filter_group)
        filter_layout.setContentsMargins(10, 18, 10, 10)
        filter_layout.setSpacing(8)

        self.calib_btn = QPushButton("🔄 背景底噪校准 (归零)")
        self.calib_btn.setObjectName("CalibBtn")
        self.calib_btn.clicked.connect(self.trigger_calibration)  # type: ignore
        filter_layout.addWidget(self.calib_btn)

        self.median_filter_cb = QCheckBox("中值滤波器 (3x3)")
        self.median_filter_cb.setChecked(True)
        filter_layout.addWidget(self.median_filter_cb)

        self.gaussian_filter_cb = QCheckBox("高斯平滑 (σ=0.5)")
        self.gaussian_filter_cb.setChecked(True)
        filter_layout.addWidget(self.gaussian_filter_cb)

        self.deadzone_cb = QCheckBox("死区阈值过滤 (门限30)")
        self.deadzone_cb.setChecked(True)
        filter_layout.addWidget(self.deadzone_cb)

        self.invert_cb = QCheckBox("反转压力方向 (压降阻值)")
        self.invert_cb.setChecked(True)
        filter_layout.addWidget(self.invert_cb)

        left_layout.addWidget(filter_group)

        # 3. 🆕 呼吸信号采集组
        breath_group = QGroupBox("🫁 呼吸信号采集")
        breath_layout = QVBoxLayout(breath_group)
        breath_layout.setContentsMargins(10, 18, 10, 10)
        breath_layout.setSpacing(8)

        breath_algo_row = QHBoxLayout()
        breath_algo_row.addWidget(QLabel("算法:"))
        self.breath_algo_combo = QComboBox()
        self.breath_algo_combo.addItems([
            "实时均值 (默认)", "ACMD分解", "VMD分解", "EMD分解", "AFD搜索",
            "VMD-MAPE", "GOA-VMD", "SMVMD", "MVMD", "Multi-ROI ICA"
        ])
        self.breath_algo_combo.currentTextChanged.connect(self.clear_algo_buffers)  # type: ignore
        breath_algo_row.addWidget(self.breath_algo_combo)
        breath_layout.addLayout(breath_algo_row)

        self.breath_bpm_label = QLabel("呼吸 BPM: --")
        self.breath_bpm_label.setStyleSheet(
            "color: #34d399; font-weight: bold; font-size: 16px;"
        )
        breath_layout.addWidget(self.breath_bpm_label)

        left_layout.addWidget(breath_group)

        # 4. 🆕 心跳信号采集组
        heartbeat_group = QGroupBox("💓 心跳信号采集")
        heartbeat_layout = QVBoxLayout(heartbeat_group)
        heartbeat_layout.setContentsMargins(10, 18, 10, 10)
        heartbeat_layout.setSpacing(8)

        hb_algo_row = QHBoxLayout()
        hb_algo_row.addWidget(QLabel("算法:"))
        self.heartbeat_algo_combo = QComboBox()
        self.heartbeat_algo_combo.addItems([
            "带通均值 (默认)", "ACMD分解", "VMD分解", "EMD分解", "VME模态追踪"
        ])
        self.heartbeat_algo_combo.currentTextChanged.connect(self.clear_algo_buffers)  # type: ignore
        hb_algo_row.addWidget(self.heartbeat_algo_combo)
        heartbeat_layout.addLayout(hb_algo_row)

        self.heartbeat_bpm_label = QLabel("心跳 BPM: --")
        self.heartbeat_bpm_label.setStyleSheet(
            "color: #f87171; font-weight: bold; font-size: 16px;"
        )
        heartbeat_layout.addWidget(self.heartbeat_bpm_label)

        left_layout.addWidget(heartbeat_group)

        # 5. 数据保存设置
        save_group = QGroupBox("💾 数据自动保存")
        save_layout = QVBoxLayout(save_group)
        save_layout.setContentsMargins(10, 18, 10, 10)
        save_layout.setSpacing(8)

        self.save_cb = QCheckBox("自动存储为 TXT 文件")
        save_layout.addWidget(self.save_cb)

        self.change_dir_btn = QPushButton("📁 更改保存文件夹")
        self.change_dir_btn.setStyleSheet("background-color: #475569;")
        self.change_dir_btn.clicked.connect(self.choose_save_directory)  # type: ignore
        save_layout.addWidget(self.change_dir_btn)

        self.dir_label = QLabel(
            f"路径: ...{self.save_path[-20:] if len(self.save_path) > 20 else self.save_path}"
        )
        self.dir_label.setToolTip(self.save_path)
        self.dir_label.setStyleSheet("color: #64748b; font-size: 10px;")
        save_layout.addWidget(self.dir_label)

        left_layout.addWidget(save_group)

        # 6. 实时状态面板
        status_group = QGroupBox("📈 实时状态")
        status_layout = QVBoxLayout(status_group)
        status_layout.setContentsMargins(10, 18, 10, 10)
        status_layout.setSpacing(6)

        self.status_label = QLabel("状态: 等待开启...")
        self.status_label.setStyleSheet("color: #10b981; font-weight: bold; font-size: 13px;")
        status_layout.addWidget(self.status_label)

        self.metrics_label = QLabel("FPS: 0 | Max: -- | Min: --")
        status_layout.addWidget(self.metrics_label)

        self.hover_label = QLabel("鼠标: 行 --, 列 -- | 压强: --")
        self.hover_label.setStyleSheet("color: #38bdf8; font-weight: bold;")
        status_layout.addWidget(self.hover_label)

        left_layout.addWidget(status_group)
        left_layout.addStretch()

        left_scroll.setWidget(left_widget)
        main_layout.addWidget(left_scroll, 1)

        # ── 右侧视图区域 (采用全屏 Tab 分离设计，不再左右分割挤压) ──
        self.main_tab_widget = QTabWidget()

        # ── Tab 1: 🔥 压力热力分布图 ──
        heatmap_tab = QWidget()
        heatmap_tab_layout = QVBoxLayout(heatmap_tab)
        heatmap_tab_layout.setContentsMargins(12, 12, 12, 12)
        heatmap_tab_layout.setSpacing(8)

        cmap_layout = QHBoxLayout()
        cmap_layout.addWidget(QLabel("调色板:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(['inferno', 'viridis', 'plasma', 'magma', 'cividis', 'turbo'])
        self.cmap_combo.setCurrentText('inferno')
        self.cmap_combo.currentTextChanged.connect(self.change_heatmap_colormap)  # type: ignore
        cmap_layout.addWidget(self.cmap_combo)

        self.smooth_render_cb = QCheckBox("平滑插值")
        self.smooth_render_cb.setChecked(True)
        self.smooth_render_cb.toggled.connect(self.toggle_smooth_rendering)  # type: ignore
        cmap_layout.addWidget(self.smooth_render_cb)
        cmap_layout.addStretch()
        heatmap_tab_layout.addLayout(cmap_layout)

        self.heatmap_view = RealTimeHeatmapView()
        self.heatmap_view.mouse_moved_signal.connect(self.display_hover_metrics)  # type: ignore
        heatmap_tab_layout.addWidget(self.heatmap_view, 1)

        self.main_tab_widget.addTab(heatmap_tab, "🔥 压力热力分布图")

        # ── Tab 2: 📈 实时生理信号波形 ──
        wave_tab = QWidget()
        wave_tab_layout = QVBoxLayout(wave_tab)
        wave_tab_layout.setContentsMargins(12, 12, 12, 12)
        wave_tab_layout.setSpacing(12)

        # 呼吸波形
        breath_wave_group = QGroupBox("🫁 实时呼吸波形")
        breath_wave_vbox = QVBoxLayout(breath_wave_group)
        breath_wave_vbox.setContentsMargins(10, 20, 10, 10)
        self.breath_plot = pg.PlotWidget()
        self.breath_plot.setMouseEnabled(x=False, y=False)  # 禁用鼠标缩放与拖拽，防止影响视图
        self.breath_plot.setMenuEnabled(False)  # 禁用右键快捷菜单
        self.breath_plot.setTitle("🫁 呼吸信号 (0.1–0.5 Hz)", color='#34d399', size='11pt')
        self.breath_plot.setBackground('#0b0f19')
        self.breath_plot.showGrid(x=True, y=True, alpha=0.3)
        self.breath_plot.setLabel('left', '幅值')
        self.breath_plot.setLabel('bottom', '采样点')
        self.breath_curve_raw = self.breath_plot.plot(
            pen=pg.mkPen('#34d399', width=2), name="实时信号"
        )
        self.breath_curve_algo = self.breath_plot.plot(
            pen=pg.mkPen('#fbbf24', width=2, style=Qt.DashLine), name="算法输出"  # type: ignore
        )
        self.breath_plot.setYRange(-30, 30)
        breath_wave_vbox.addWidget(self.breath_plot)
        wave_tab_layout.addWidget(breath_wave_group)

        # 心跳波形
        heartbeat_wave_group = QGroupBox("💓 实时心跳波形")
        heartbeat_wave_vbox = QVBoxLayout(heartbeat_wave_group)
        heartbeat_wave_vbox.setContentsMargins(10, 20, 10, 10)
        self.heartbeat_plot = pg.PlotWidget()
        self.heartbeat_plot.setMouseEnabled(x=False, y=False)  # 禁用鼠标缩放与拖拽，防止影响视图
        self.heartbeat_plot.setMenuEnabled(False)  # 禁用右键快捷菜单
        self.heartbeat_plot.setTitle("💓 心跳信号 (0.8–2.2 Hz)", color='#f87171', size='11pt')
        self.heartbeat_plot.setBackground('#0b0f19')
        self.heartbeat_plot.showGrid(x=True, y=True, alpha=0.3)
        self.heartbeat_plot.setLabel('left', '幅值')
        self.heartbeat_plot.setLabel('bottom', '采样点')
        self.heartbeat_curve_raw = self.heartbeat_plot.plot(
            pen=pg.mkPen('#f87171', width=2), name="实时信号"
        )
        self.heartbeat_curve_algo = self.heartbeat_plot.plot(
            pen=pg.mkPen('#fbbf24', width=2, style=Qt.DashLine), name="算法输出"  # type: ignore
        )
        self.heartbeat_plot.setYRange(-10, 10)
        heartbeat_wave_vbox.addWidget(self.heartbeat_plot)
        wave_tab_layout.addWidget(heartbeat_wave_group)

        self.main_tab_widget.addTab(wave_tab, "📈 实时生理信号波形")

        # ── Tab 3: 📊 原始数据矩阵网格 ──
        table_tab = QWidget()
        table_tab_layout = QVBoxLayout(table_tab)
        table_tab_layout.setContentsMargins(12, 12, 12, 12)
        table_tab_layout.setSpacing(8)

        table_options = QHBoxLayout()
        self.table_view_mode_cb = QCheckBox("仅显示去噪后的压力净值")
        self.table_view_mode_cb.setChecked(True)
        table_options.addWidget(self.table_view_mode_cb)
        table_tab_layout.addLayout(table_options)

        self.table_widget = QTableWidget(40, 40)
        self.setup_table_widget()
        table_tab_layout.addWidget(self.table_widget)

        self.main_tab_widget.addTab(table_tab, "📊 原始数据矩阵网格 (ADC Counts)")

        main_layout.addWidget(self.main_tab_widget, 5)

    def setup_table_widget(self):
        """
        初始化 40x40 的数据格
        """
        h_header = self.table_widget.horizontalHeader()
        v_header = self.table_widget.verticalHeader()
        if h_header is not None:
            h_header.setDefaultSectionSize(26)
            h_header.setStyleSheet("font-size: 7px;")
        if v_header is not None:
            v_header.setDefaultSectionSize(18)
            v_header.setStyleSheet("font-size: 7px;")
        self.table_widget.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_widget.setStyleSheet("font-size: 7px; font-family: Consolas;")

        for r in range(40):
            for c in range(40):
                item = QTableWidgetItem("0")
                item.setTextAlignment(Qt.AlignCenter)  # type: ignore
                self.table_widget.setItem(r, c, item)

    # ══════════════════════════════════════════════
    # UI 交互方法
    # ══════════════════════════════════════════════
    def clear_algo_buffers(self):
        """算法更换时清空历史累加器和权重，避免旧算法数据残留导致过渡跳变"""
        self.breath_algo_accum.fill(0.0)
        self.breath_algo_weight.fill(1e-6)
        self.heartbeat_algo_accum.fill(0.0)
        self.heartbeat_algo_weight.fill(1e-6)

    def refresh_ports(self):
        self.port_combo.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo.addItems(ports)
        if not ports:
            self.port_combo.addItem("无可用COM端口")

    def change_heatmap_colormap(self, text):
        self.heatmap_view.change_colormap(text)

    def toggle_smooth_rendering(self, checked):
        self.heatmap_view.img.setOpts(smooth=checked)

    def choose_save_directory(self):
        path = QFileDialog.getExistingDirectory(self, "选择保存文件夹", self.save_path)
        if path:
            self.save_path = path
            display_path = path[-20:] if len(path) > 20 else path
            self.dir_label.setText(f"路径: ...{display_path}")
            self.dir_label.setToolTip(path)

    def trigger_calibration(self):
        self.calibrating = True
        self.status_label.setText("正在执行零点校准...")
        self.status_label.setStyleSheet("color: #8b5cf6; font-weight: bold;")

    # ══════════════════════════════════════════════
    # 采集启动 / 停止
    # ══════════════════════════════════════════════
    def start_acquisition(self):
        # 1. 判断是否开启自动文件保存
        if self.save_cb.isChecked():
            try:
                filename = datetime.now().strftime("%Y%m%d_%H%M%S_40x40.txt")
                self.file_path = os.path.join(self.save_path, filename)
                self.file_handle = open(self.file_path, 'w', encoding='utf-8')
                self.is_recording = True
                print(f"创建数据记录文件: {self.file_path}")
            except Exception as e:
                QMessageBox.critical(self, "文件保存错误", f"无法创建保存文件: {e}")
                return

        # 2. 配置串口线程
        self.serial_thread.simulate = self.simulate_cb.isChecked()
        self.serial_thread.port = self.port_combo.currentText()
        try:
            self.serial_thread.baudrate = int(self.baud_combo.currentText())
        except:
            self.serial_thread.baudrate = 115200

        self.serial_thread.endianness = (
            'little' if 'Little' in self.endian_combo.currentText() else 'big'
        )

        # 3. 🆕 重置信号缓冲区
        self.breath_buffer = np.zeros(SIGNAL_BUFFER_SIZE)
        self.heartbeat_buffer = np.zeros(SIGNAL_BUFFER_SIZE)
        self.frame_buffer = []
        self.breath_signal_smoothed = np.zeros(SIGNAL_BUFFER_SIZE)
        self.heartbeat_signal_filtered = np.zeros(SIGNAL_BUFFER_SIZE)
        self.breath_bpm = 0.0
        self.heartbeat_bpm = 0.0
        self.signal_frame_counter = 0

        # 4. 启动线程
        self.serial_thread.start()

        # 5. 🆕 启动算法定时器
        self.algo_timer.start(ALGO_INTERVAL_MS)

        # 6. 更新控件状态
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.simulate_cb.setEnabled(False)
        self.port_combo.setEnabled(False)
        self.baud_combo.setEnabled(False)
        self.endian_combo.setEnabled(False)

        mode_str = "模拟测试" if self.simulate_cb.isChecked() else "物理串口"
        self.status_label.setText(f"采集运行中... [{mode_str}]")
        self.status_label.setStyleSheet("color: #10b981; font-weight: bold;")

    def stop_acquisition(self):
        self.serial_thread.stop()
        self.serial_thread.wait()

        # 🆕 停止算法定时器
        self.algo_timer.stop()

        # 关闭记录文件
        if self.file_handle:
            self.file_handle.flush()
            self.file_handle.close()
            self.file_handle = None
        self.is_recording = False

        # 控件状态还原
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.simulate_cb.setEnabled(True)
        self.port_combo.setEnabled(True)
        self.baud_combo.setEnabled(True)
        self.endian_combo.setEnabled(True)

        self.status_label.setText("采集已停止。")
        self.status_label.setStyleSheet("color: #ef4444; font-weight: bold;")
        self.metrics_label.setText("FPS: 0 | Max: -- | Min: --")
        self.breath_bpm_label.setText("呼吸 BPM: --")
        self.heartbeat_bpm_label.setText("心跳 BPM: --")

    # ══════════════════════════════════════════════
    # 数据接收与处理（核心管道）
    # ══════════════════════════════════════════════
    def on_data_received(self, matrix):
        """
        接收从串口线程解码后的 40x40 原始矩阵，执行完整处理管道：
        校准 → 滤波 → 热力图 → 信号提取 → 数据保存
        """
        raw_matrix = matrix.copy()

        # ── 1. 零点背景校准 ──
        if self.calibrating:
            self.base_matrix = raw_matrix.copy()
            self.is_calibrated = True
            self.calibrating = False
            self.status_label.setText("零点校准已完成")
            self.status_label.setStyleSheet("color: #10b981; font-weight: bold;")

        # ── 2. 去除底噪与压感方向反转 ──
        if self.is_calibrated:
            if self.invert_cb.isChecked():
                processed = self.base_matrix.astype(np.int32) - raw_matrix.astype(np.int32)
            else:
                processed = raw_matrix.astype(np.int32) - self.base_matrix.astype(np.int32)
            processed[processed < 0] = 0
            processed = processed.astype(np.uint16)
        else:
            if self.invert_cb.isChecked():
                processed = np.clip(
                    4095 - raw_matrix.astype(np.int32), 0, 4095
                ).astype(np.uint16)
            else:
                processed = raw_matrix.copy()

        # ── 3. 死区阈值门限过滤 ──
        if self.deadzone_cb.isChecked():
            processed[processed < 30] = 0

        # ── 4. 数据自动记录 ──
        if self.is_recording and self.file_handle:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            row_data_str = " ".join(map(str, processed.flatten()))
            self.file_handle.write(f"{timestamp} {row_data_str}\n")
            if self.frame_count % 15 == 0:
                self.file_handle.flush()

        # ── 5. 空间滤波（用于可视化） ──
        vis_matrix = processed.astype(np.float64)
        if self.median_filter_cb.isChecked():
            vis_matrix = median_filter(vis_matrix, size=3)
        if self.gaussian_filter_cb.isChecked():
            vis_matrix = gaussian_filter(vis_matrix, sigma=0.5)

        # ── 6. 更新热力图 ──
        self.heatmap_view.draw_matrix(vis_matrix)

        # ── 7. 暂存当前矩阵供表格显示 ──
        if self.table_view_mode_cb.isChecked():
            self.current_processed_matrix = processed.copy()
        else:
            self.current_processed_matrix = raw_matrix.copy()
        
        # ── 8. 🆕 信号提取（降采样至 ~10 Hz） ──
        self.signal_frame_counter += 1
        if self.signal_frame_counter % self.downsample_factor == 0:
            self._extract_signals(vis_matrix)
            self.frame_buffer.append(vis_matrix.copy())
            if len(self.frame_buffer) > SIGNAL_BUFFER_SIZE:
                self.frame_buffer.pop(0)

        # ── 9. 🆕 实时波形更新 ──
        self._update_waveforms()

        # ── 10. FPS 与统计 ──
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            fps = self.frame_count / (now - self.last_fps_time)
            max_val = np.max(processed)
            min_val = np.min(processed)
            self.downsample_factor = max(1, int(fps / TARGET_SAMPLE_RATE))
            self.metrics_label.setText(
                f"FPS: {fps:.1f} | Max: {max_val} | Min: {min_val}"
            )
            self.frame_count = 0
            self.last_fps_time = now

    # ══════════════════════════════════════════════
    # 🆕 信号提取方法
    # ══════════════════════════════════════════════
    def _extract_signals(self, vis_matrix):
        """
        从当前处理后的 40x40 帧中提取呼吸和心跳信号标量值，
        追加到滚动缓冲区。
        """
        # 呼吸信号：压力 > 100 的像素均值
        active_breath = vis_matrix[vis_matrix > 100]
        breath_val = (
            np.mean(active_breath) if active_breath.size > 0
            else np.mean(vis_matrix[vis_matrix > 0]) if np.any(vis_matrix > 0)
            else 0.0
        )

        # 心跳信号：使用更敏感的阈值（压力 > 50），捕获微弱高频波动
        active_hb = vis_matrix[vis_matrix > 50]
        heartbeat_val = (
            np.mean(active_hb) if active_hb.size > 0
            else np.mean(vis_matrix[vis_matrix > 0]) if np.any(vis_matrix > 0)
            else 0.0
        )

        # 滚动缓冲区更新
        self.breath_buffer = np.roll(self.breath_buffer, -1)
        self.breath_buffer[-1] = breath_val

        self.heartbeat_buffer = np.roll(self.heartbeat_buffer, -1)
        self.heartbeat_buffer[-1] = heartbeat_val

        # 随着时序推移，同步滚动算法估计累加器和权重，末位开辟新空间初始化为0
        self.breath_algo_accum = np.roll(self.breath_algo_accum, -1)
        self.breath_algo_accum[-1] = 0.0
        self.breath_algo_weight = np.roll(self.breath_algo_weight, -1)
        self.breath_algo_weight[-1] = 1e-6

        self.heartbeat_algo_accum = np.roll(self.heartbeat_algo_accum, -1)
        self.heartbeat_algo_accum[-1] = 0.0
        self.heartbeat_algo_weight = np.roll(self.heartbeat_algo_weight, -1)
        self.heartbeat_algo_weight[-1] = 1e-6

    def _update_waveforms(self):
        """
        实时更新呼吸和心跳波形曲线。
        """
        # ── 呼吸波形：去趋势 + 平滑 ──
        if np.any(self.breath_buffer):
            breath_detrend = self.breath_buffer - np.mean(self.breath_buffer)
            self.breath_signal_smoothed = smooth_signal(breath_detrend, window=11, polyorder=3)
            self.breath_curve_raw.setData(self.breath_signal_smoothed)
            # 自适应 Y 轴范围
            bmax = np.max(np.abs(self.breath_signal_smoothed))
            self.breath_plot.setYRange(-max(bmax * 1.2, 0.1), max(bmax * 1.2, 0.1))

        # ── 心跳波形：去趋势 + 带通滤波预览 ──
        if np.any(self.heartbeat_buffer):
            hb_detrend = self.heartbeat_buffer - np.mean(self.heartbeat_buffer)
            if len(hb_detrend) >= 16:
                try:
                    from algorithms.base import butter_bandpass_filter
                    hb_filtered = butter_bandpass_filter(
                        hb_detrend, lowcut=HEARTBEAT_LOWCUT, highcut=HEARTBEAT_HIGHCUT,
                        fs=TARGET_SAMPLE_RATE, order=3
                    )
                    self.heartbeat_signal_filtered = smooth_signal(hb_filtered, window=7, polyorder=2)
                except Exception:
                    self.heartbeat_signal_filtered = smooth_signal(hb_detrend, window=7, polyorder=2)
            else:
                self.heartbeat_signal_filtered = hb_detrend
            self.heartbeat_curve_raw.setData(self.heartbeat_signal_filtered)
            hmax = np.max(np.abs(self.heartbeat_signal_filtered))
            self.heartbeat_plot.setYRange(-max(hmax * 1.25, 0.05), max(hmax * 1.25, 0.05))

    # ══════════════════════════════════════════════
    # 🆕 算法执行（每秒触发一次）
    # ══════════════════════════════════════════════
    def run_algorithms(self):
        """滑动窗口算法执行（每秒触发一次，窗口=15s）"""
        # ── 呼吸算法 ──
        breath_algo = self.breath_algo_combo.currentText()
        breath_raw = self.breath_buffer - np.mean(self.breath_buffer)

        # 获取 3D 帧序列用于空间-时间算法 (至少积累 10 秒，即 112 帧)
        frames_3d = np.array(self.frame_buffer) if len(self.frame_buffer) >= 112 else None

        if np.max(breath_raw) - np.min(breath_raw) < 0.05:
            self.breath_bpm = 0.0
            breath_result = np.zeros_like(breath_raw)
        else:
            try:
                if breath_algo == "实时均值 (默认)":
                    breath_result = extract_breath_mean(breath_raw)
                elif breath_algo == "ACMD分解":
                    breath_result = extract_breath_acmd(breath_raw, fs=TARGET_SAMPLE_RATE)
                elif breath_algo == "VMD分解":
                    breath_result = extract_breath_vmd(breath_raw, fs=TARGET_SAMPLE_RATE)
                elif breath_algo == "EMD分解":
                    breath_result = extract_breath_emd(breath_raw, fs=TARGET_SAMPLE_RATE)
                elif breath_algo == "AFD搜索":
                    breath_result = extract_breath_afd(breath_raw, fs=TARGET_SAMPLE_RATE)
                elif breath_algo == "VMD-MAPE":
                    inp = frames_3d if frames_3d is not None else breath_raw
                    breath_result = extract_breath_vmd_mape(inp, fs=TARGET_SAMPLE_RATE)
                elif breath_algo == "GOA-VMD":
                    inp = frames_3d if frames_3d is not None else breath_raw
                    breath_result = extract_breath_goa_vmd(inp, fs=TARGET_SAMPLE_RATE)
                elif breath_algo == "SMVMD":
                    if frames_3d is not None:
                        breath_result = extract_breath_smvmd(frames_3d, fs=TARGET_SAMPLE_RATE)
                    else:
                        breath_result = extract_breath_vmd(breath_raw, fs=TARGET_SAMPLE_RATE)
                elif breath_algo == "MVMD":
                    if frames_3d is not None:
                        breath_result = extract_breath_mvmd(frames_3d, fs=TARGET_SAMPLE_RATE)
                    else:
                        breath_result = extract_breath_vmd(breath_raw, fs=TARGET_SAMPLE_RATE)
                elif breath_algo == "Multi-ROI ICA":
                    if frames_3d is not None:
                        breath_result = extract_breath_multi_roi_ica(frames_3d, fs=TARGET_SAMPLE_RATE)
                    else:
                        breath_result = extract_breath_mean(breath_raw)
                else:
                    breath_result = extract_breath_mean(breath_raw)
                
                # FPR 呼吸 BPM：主波间距 ≥ 1.5s（最快40次/min）
                self.breath_bpm = calculate_bpm_fpr(
                    breath_result, fs=TARGET_SAMPLE_RATE, min_dist_s=1.5
                )
            except Exception as e:
                print(f"[呼吸算法异常] {e}")
                breath_result = extract_breath_mean(breath_raw)
                self.breath_bpm = calculate_bpm_fpr(breath_result, fs=TARGET_SAMPLE_RATE, min_dist_s=1.5)

        if np.any(breath_result):
            # 将当前窗口计算的 15s 信号，以正弦窗权重叠加到历史累加器中
            self.breath_algo_accum += breath_result * self.window_weights
            self.breath_algo_weight += self.window_weights

        # 计算所有交集窗口的加权平均值
        blended_breath = self.breath_algo_accum / self.breath_algo_weight
        
        # 动态自适应方差匹配缩放，使算法提取波形在视觉上与原始绿线完美重合匹配，极度明显
        std_raw = np.std(self.breath_signal_smoothed)
        std_res = np.std(blended_breath)
        if std_res > 1e-6:
            blended_breath = blended_breath * (std_raw / std_res) * 0.85
        self.breath_curve_algo.setData(blended_breath)

        # ── 心跳算法 ──
        heartbeat_algo = self.heartbeat_algo_combo.currentText()
        heartbeat_raw = self.heartbeat_buffer - np.mean(self.heartbeat_buffer)
        hb_result = np.zeros_like(heartbeat_raw)

        if np.max(heartbeat_raw) - np.min(heartbeat_raw) >= 0.02:
            try:
                if heartbeat_algo == "带通均值 (默认)":
                    hb_result = extract_heartbeat_mean(heartbeat_raw, fs=TARGET_SAMPLE_RATE)
                elif heartbeat_algo == "ACMD分解":
                    hb_result = extract_heartbeat_acmd(heartbeat_raw, fs=TARGET_SAMPLE_RATE)
                elif heartbeat_algo == "VMD分解":
                    hb_result = extract_heartbeat_vmd(heartbeat_raw, fs=TARGET_SAMPLE_RATE)
                elif heartbeat_algo == "EMD分解":
                    hb_result = extract_heartbeat_emd(heartbeat_raw, fs=TARGET_SAMPLE_RATE)
                elif heartbeat_algo == "VME模态追踪":
                    inp = frames_3d if frames_3d is not None else heartbeat_raw
                    hb_result = extract_heartbeat_vme(inp, fs=TARGET_SAMPLE_RATE)
                else:
                    hb_result = extract_heartbeat_mean(heartbeat_raw, fs=TARGET_SAMPLE_RATE)
                # FPR 心跳 BPM：主波间距 ≥ 0.4s（最快150次/min）
                self.heartbeat_bpm = calculate_bpm_fpr(
                    hb_result, fs=TARGET_SAMPLE_RATE, min_dist_s=0.4
                )
            except Exception as e:
                print(f"[心跳算法异常] {e}")
                hb_result = extract_heartbeat_mean(heartbeat_raw, fs=TARGET_SAMPLE_RATE)
                self.heartbeat_bpm = calculate_bpm_fpr(hb_result, fs=TARGET_SAMPLE_RATE, min_dist_s=0.4)
        else:
            self.heartbeat_bpm = 0.0

        if np.any(hb_result):
            # 将当前窗口计算的 15s 信号，以正弦窗权重叠加到历史累加器中
            self.heartbeat_algo_accum += hb_result * self.window_weights
            self.heartbeat_algo_weight += self.window_weights

        # 计算所有交集窗口的加权平均值
        blended_hb = self.heartbeat_algo_accum / self.heartbeat_algo_weight

        std_raw = np.std(self.heartbeat_signal_filtered)
        std_res = np.std(blended_hb)
        if std_res > 1e-6:
            blended_hb = blended_hb * (std_raw / std_res) * 0.85
        self.heartbeat_curve_algo.setData(blended_hb)

        self.breath_bpm_label.setText(f"呼吸 BPM: {self.breath_bpm:.1f}")
        self.heartbeat_bpm_label.setText(f"心跳 BPM: {self.heartbeat_bpm:.1f}")

    # ══════════════════════════════════════════════
    # 表格更新与其他
    # ══════════════════════════════════════════════
    def update_raw_table(self):
        """
        分频刷新 40x40 原始/去噪数据网格，确保界面零卡顿
        """
        if not self.serial_thread.isRunning():
            return

        self.table_widget.setUpdatesEnabled(False)
        matrix = self.current_processed_matrix
        for r in range(40):
            for c in range(40):
                val = int(matrix[r, c])
                item = self.table_widget.item(r, c)
                if item is not None:
                    item.setText(str(val))
        self.table_widget.setUpdatesEnabled(True)

    def display_hover_metrics(self, r, c, val):
        self.hover_label.setText(
            f"鼠标: 行 {r:02d}, 列 {c:02d} | 压强: {val:.1f}"
        )

    def on_serial_error(self, error_msg):
        self.stop_acquisition()
        QMessageBox.critical(self, "串口采集异常", f"检测到串口错误已自动断开：\n{error_msg}")

    def closeEvent(self, a0):
        self.stop_acquisition()
        a0.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
