import sys
import serial
import serial.tools.list_ports
import numpy as np
import time
import os
from datetime import datetime

# 修复 conda 环境下 PyQt5 DLL 找不到的问题
_conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)
_qt_dll_path = os.path.join(_conda_prefix, 'Library', 'bin')
if os.path.isdir(_qt_dll_path) and _qt_dll_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _qt_dll_path + os.pathsep + os.environ.get('PATH', '')

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QComboBox,
                             QCheckBox, QFileDialog, QMessageBox, QGroupBox,
                             QSplitter)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QRectF
from PyQt5.QtGui import QPainter, QColor, QFont
import pyqtgraph as pg

# 导入算法
import algorithms.base as algo
import algorithms.vmd_extract as vmd_algo
import algorithms.emd_extract as emd_algo
import algorithms.afd_extract as afd_algo
from preprocess import Preprocessor

MATRIX_DIM = 32


class SerialThread(QThread):
    data_received = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.port = ''
        self.baudrate = 460800
        self.running = False

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.ser.reset_input_buffer()
            self.ser.write(b'1')
            self.running = True
            while self.running:
                if self.ser.in_waiting >= 2050:
                    header = self.ser.read(2)
                    if header == b'\xaa\x55':
                        raw_data = self.ser.read(2048)
                        data = np.frombuffer(raw_data, dtype=np.uint16)
                        if data.size == 1024:
                            self.data_received.emit(data.reshape((MATRIX_DIM, MATRIX_DIM)).copy())
                    else:
                        self.ser.read(1)
                else:
                    time.sleep(0.005)
            self.ser.write(b'2')
            self.ser.close()
        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        self.running = False


class HeatmapWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.matrix = np.zeros((MATRIX_DIM, MATRIX_DIM))
        self.setMinimumSize(400, 400)

    def update_matrix(self, matrix):
        self.matrix = matrix
        self.update()

    def clamp(self, v):
        """将颜色分量限制在 0-1 之间"""
        return max(0.0, min(1.0, v))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        # 计算正方形区域
        side = min(self.width(), self.height())
        offset_x = (self.width() - side) / 2
        offset_y = (self.height() - side) / 2
        cell_size = side / MATRIX_DIM

        for r in range(MATRIX_DIM):
            for c in range(MATRIX_DIM):
                val = self.matrix[r, c]
                # 归一化映射 (对应 process_data 中的 800-2500 范围)
                norm = max(0.0, min(1.0, val / 300.0))

                # Jet 算法映射：深蓝 -> 天蓝 -> 绿 -> 黄 -> 红
                r_val = self.clamp(min(4 * norm - 1.5, -4 * norm + 4.5))
                g_val = self.clamp(min(4 * norm - 0.5, -4 * norm + 3.5))
                b_val = self.clamp(min(4 * norm + 0.5, -4 * norm + 2.5))

                color = QColor(int(r_val * 255), int(g_val * 255), int(b_val * 255))
                painter.fillRect(QRectF(offset_x + c * cell_size,
                                        offset_y + r * cell_size,
                                        cell_size + 0.5, cell_size + 0.5), color)


class RealTimeMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("压力矩阵呼吸监测系统")
        self.resize(1500, 900)

        # 默认路径
        self.save_path = r"D:\1\bs\new_CUSHION\data"
        if not os.path.exists(self.save_path):
            try:
                os.makedirs(self.save_path)
            except:
                self.save_path = os.getcwd()

        # 数据变量
        self.breath_buffer = np.zeros(200)
        self.base_matrix = np.zeros((32, 32))
        self.is_calibrated = False
        self.is_recording = False
        self.file_handle = None
        self.count = 0
        self.last_time = time.time()

        self.preprocessor = Preprocessor(deadzone=30)

        self.setup_ui()
        self.serial_thread = SerialThread()
        self.serial_thread.data_received.connect(self.process_data)
        self.serial_thread.error_occurred.connect(self.handle_error)

        self.is_label_recording = False  # 标注录制状态
        self.label_data_cache = []  # 内存数据缓冲区

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # 1. 主水平分割器 (左侧面板 | 右侧波形)
        self.main_h_splitter = QSplitter(Qt.Horizontal)

        # --- 左侧部分容器 ---
        left_widget = QWidget()
        left_container_layout = QVBoxLayout(left_widget)
        left_container_layout.setContentsMargins(0, 0, 0, 0)

        # 2. 左侧垂直分割器 (左上选项 | 左下热力图)
        self.left_v_splitter = QSplitter(Qt.Vertical)

        # 左上选项区
        self.top_options = QWidget()
        top_layout = QVBoxLayout(self.top_options)

        # 连接控制
        g1 = QGroupBox("1. 硬件连接")
        l1 = QVBoxLayout(g1)
        self.port_box = QComboBox()
        self.port_box.addItems([p.device for p in serial.tools.list_ports.comports()])
        self.btn_start = QPushButton("▶ 开始采集")
        self.btn_stop = QPushButton("■ 停止")
        self.btn_calib = QPushButton("🔄 校准")
        self.btn_start.clicked.connect(self.start_serial)
        self.btn_stop.clicked.connect(self.stop_serial)
        self.btn_calib.clicked.connect(lambda: setattr(self, 'is_calibrated', False))
        l1.addWidget(self.port_box);
        l1.addWidget(self.btn_start);
        l1.addWidget(self.btn_stop);
        l1.addWidget(self.btn_calib)
        top_layout.addWidget(g1)


        # 保存设置
        g2 = QGroupBox("2. 保存设置")
        l2 = QVBoxLayout(g2)
        self.cb_save = QCheckBox("自动保存数据")
        self.btn_path = QPushButton("📁 选择路径")
        self.btn_path.clicked.connect(self.select_path)
        self.lbl_path = QLabel(f"路径: {self.save_path}")
        self.lbl_path.setWordWrap(True)
        l2.addWidget(self.cb_save);
        l2.addWidget(self.btn_path);
        l2.addWidget(self.lbl_path)
        top_layout.addWidget(g2)

        # 坐姿数据采集
        label_group = QGroupBox("3. 坐姿标注模式")
        label_lay = QVBoxLayout(label_group)

        self.label_combo = QComboBox()
        self.label_combo.addItems(["0-正常", "1-前倾", "2-后仰", "3-左倾", "4-右倾", "5-离座"])

        self.btn_capture = QPushButton("⏺ 开始录制标注数据")
        self.btn_capture.setStyleSheet("background-color: #27ae60; color: white;")
        self.btn_capture.clicked.connect(self.toggle_label_recording)

        label_lay.addWidget(QLabel("选择当前姿态:"))
        label_lay.addWidget(self.label_combo)
        label_lay.addWidget(self.btn_capture)
        top_layout.addWidget(label_group)

        # 呼吸算法选择
        algo_group = QGroupBox("4. 呼吸算法选择")
        algo_lay = QVBoxLayout(algo_group)
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["实时均值 (默认)", "VMD分解", "EMD分解", "AFD搜索"])
        algo_lay.addWidget(self.algo_combo)
        top_layout.addWidget(algo_group)  # 添加到左侧控制面板


        # 状态
        self.lbl_status = QLabel("BPM: 0.0\nFPS: 0")
        self.lbl_status.setStyleSheet("font-size: 18px; color: blue; font-weight: bold;")
        top_layout.addWidget(self.lbl_status)
        top_layout.addStretch()

        # 左下热力图区
        self.heatmap = HeatmapWidget()

        # 组装左侧分割器
        self.left_v_splitter.addWidget(self.top_options)
        self.left_v_splitter.addWidget(self.heatmap)
        self.left_v_splitter.setStretchFactor(1, 2)  # 热力图占更多空间

        # 右侧波形图
        self.pw = pg.PlotWidget(title="实时呼吸信号")
        self.pw.setBackground('w')
        self.pw.showGrid(x=True, y=True)
        self.curve = self.pw.plot(pen=pg.mkPen('r', width=2))
        self.pw.setYRange(-20, 20)

        # 组装水平分割器
        self.main_h_splitter.addWidget(self.left_v_splitter)
        self.main_h_splitter.addWidget(self.pw)
        self.main_h_splitter.setStretchFactor(1, 3)  # 波形图占主要空间

        main_layout.addWidget(self.main_h_splitter)

    def select_path(self):
        p = QFileDialog.getExistingDirectory(self, "选择路径", self.save_path)
        if p: self.save_path = p; self.lbl_path.setText(f"路径: {p}")

    def start_serial(self):
        if self.cb_save.isChecked():
            fn = os.path.join(self.save_path, datetime.now().strftime("%Y%m%d_%H%M%S.txt"))
            self.file_handle = open(fn, 'w');
            self.is_recording = True
        self.serial_thread.port = self.port_box.currentText()
        self.serial_thread.start()
        self.btn_start.setEnabled(False)

    def stop_serial(self):
        self.serial_thread.stop()
        if self.file_handle: self.file_handle.close(); self.file_handle = None
        self.is_recording = False;
        self.btn_start.setEnabled(True)

    def process_data(self, matrix):
        # --- [1. 基础预处理与 heat 脚本同步逻辑] ---
        if not self.is_calibrated:
            self.base_matrix = matrix.copy()
            self.is_calibrated = True
            return

        processed = np.maximum(0, matrix.astype(np.int32) - self.base_matrix.astype(np.int32))
        processed[processed < 30] = 0

        from scipy.ndimage import median_filter, gaussian_filter
        clipped = np.clip(processed, 0, 500)
        f_matrix = median_filter(clipped, size=3)
        f_matrix = gaussian_filter(f_matrix, sigma=0.5)

        if self.is_label_recording:
            self.label_data_cache.append(clipped.copy())

        self.current_clean_frame = f_matrix.copy()
        self.heatmap.update_matrix(f_matrix)

        # --- [2. 新增：左右分区特征提取] ---
        # 矩阵切割：f_matrix 形状为 (32, 32)
        left_part = f_matrix[:, :16]
        right_part = f_matrix[:, 16:]

        # 提取特征：平均压力与最大压力
        avg_l, max_l = np.mean(left_part), np.max(left_part)
        avg_r, max_r = np.mean(right_part), np.max(right_part)

        # 计算左右平衡度 (Balance Ratio)
        total_avg = avg_l + avg_r
        balance = (avg_l / total_avg * 100) if total_avg > 0 else 50.0

        # 计算压力中心 (COP)
        def get_cop(part_matrix):
            total_sum = np.sum(part_matrix)
            if total_sum < 10: return 0.0, 0.0  # 压力过小时归零
            h, w = part_matrix.shape
            iy, ix = np.indices((h, w))
            cop_x = np.sum(ix * part_matrix) / total_sum
            cop_y = np.sum(iy * part_matrix) / total_sum
            return cop_x, cop_y

        cop_lx, cop_ly = get_cop(left_part)
        cop_rx, cop_ry = get_cop(right_part)


        # --- [3. 更新状态显示] ---
        # 呼吸提取逻辑保持原样
        # 1. 基础信号提取（作为所有算法的输入）
        active = f_matrix[f_matrix > 100]
        val = np.mean(active) if active.size > 0 else np.mean(f_matrix[f_matrix > 0]) if np.any(f_matrix > 0) else 0
        self.breath_buffer = np.roll(self.breath_buffer, -1)
        self.breath_buffer[-1] = val

        display_raw = self.breath_buffer - np.mean(self.breath_buffer)
        realtime_smooth = algo.smooth_signal(display_raw)  # 调用 base.py 的平滑
        self.curve.setData(realtime_smooth)  # 每一帧都绘制，波形就回来了

        # 2. 定时执行所选算法 (每 1 秒更新一次算法结果)
        self.count += 1
        now = time.time()
        if now - self.last_time >= 1.0:
            if np.max(self.breath_buffer) - np.min(self.breath_buffer) < 1.0:
                bpm = 0.0
                final_signal = np.zeros_like(self.breath_buffer)
            else:
                current_algo = self.algo_combo.currentText()
                # 准备当前缓存中的帧序列 (N, 32, 32)，假设你维护了一个帧缓冲区
                # 如果没有帧缓冲区，复杂算法将作用于一维 self.breath_buffer

                if current_algo == "实时均值 (默认)":
                    display_signal = self.breath_buffer - np.mean(self.breath_buffer)
                    final_signal = algo.smooth_signal(display_signal)
                    bpm = algo.calculate_bpm(final_signal)

                elif current_algo == "VMD分解":
                    # 直接对缓冲区的一维信号进行 VMD
                    final_signal = vmd_algo.extract_respiration(self.breath_buffer)
                    bpm = algo.calculate_bpm(final_signal)
                elif current_algo == "EMD分解":
                    final_signal = emd_algo.extract_respiration(self.breath_buffer)
                    bpm = algo.calculate_bpm(final_signal)
                elif current_algo == "AFD搜索":
                    final_signal = afd_algo.extract_respiration(self.breath_buffer)
                    bpm = algo.calculate_bpm(final_signal)
            # 在状态栏增加左右特征显示
            status_text = (
                f"BPM: {bpm:.1f} | FPS: {int(self.count / (now - self.last_time))}\n"
                f"左Peak: {max_l:.0f} | 右Peak: {max_r:.0f}\n"
                f"左COP: ({cop_lx:.1f}, {cop_ly:.1f}) | 右COP: ({cop_rx:.1f}, {cop_ry:.1f})"
            )
            self.lbl_status.setText(status_text)
            self.count, self.last_time = 0, now

    def capture_and_label(self):
        """将当前帧和对应标签保存到 dataset 文件夹"""
        if not hasattr(self, 'current_clean_frame'):
            QMessageBox.warning(self, "错误", "尚未开始采集数据")
            return

        # 1. 创建数据集目录
        ds_path = os.path.join(os.getcwd(), "dataset")
        if not os.path.exists(ds_path): os.makedirs(ds_path)

        # 2. 获取当前选中的标签
        label_idx = self.label_combo.currentIndex()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # 3. 保存为 npy 文件，文件名包含标签信息
        filename = f"pose_{label_idx}_{timestamp}.npy"
        filepath = os.path.join(ds_path, filename)

        # 保存当前正在显示的 clean_f
        np.save(filepath, self.current_clean_frame)

        # UI反馈
        self.lbl_status.setText(f"已保存标签 {label_idx}\n到数据集")

    def toggle_label_recording(self):
        """切换标注录制状态"""
        if not self.is_label_recording:
            # 检查是否已开启串口采集
            if not self.serial_thread.isRunning():
                QMessageBox.warning(self, "警告", "请先开启硬件连接采集数据")
                return

            # 开始录制
            self.is_label_recording = True
            self.label_data_cache = []  # 清空旧缓冲区
            self.btn_capture.setText("⏹ 停止录制并保存")
            self.btn_capture.setStyleSheet("background-color: #c0392b; color: white;")
            self.lbl_status.setText(f"正在录制姿态 {self.label_combo.currentText()}...")
        else:
            # 停止录制并保存
            self.is_label_recording = False
            self.btn_capture.setText("⏺ 开始录制标注数据")
            self.btn_capture.setStyleSheet("background-color: #27ae60; color: white;")
            self.save_labeled_batch()

    def save_labeled_batch(self):
        """将内存中的序列保存为单个大文件"""
        if not self.label_data_cache:
            QMessageBox.information(self, "提示", "未采集到有效数据")
            return

        ds_path = os.path.join(os.getcwd(), "dataset")
        if not os.path.exists(ds_path): os.makedirs(ds_path)

        # 获取标签索引
        label_idx = self.label_combo.currentIndex()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        count = len(self.label_data_cache)

        # 文件命名格式：标签_样本数_时间.npy
        filename = f"pose_{label_idx}_samples_{count}_{timestamp}.npy"
        filepath = os.path.join(ds_path, filename)

        # 将列表转换为 (N, 32, 32) 数组并保存
        final_array = np.array(self.label_data_cache)
        np.save(filepath, final_array)

        QMessageBox.information(self, "成功", f"录制结束！\n保存了 {count} 帧数据至：\n{filename}")
        self.label_data_cache = []  # 清空内存



    def handle_error(self, msg):
        self.stop_serial();
        QMessageBox.critical(self, "串口错误", msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("QSplitter::handle { background-color: #cccccc; height: 2px; width: 2px; }")
    win = RealTimeMainWindow()
    win.show()
    sys.exit(app.exec_())