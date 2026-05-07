import sys
import serial
import serial.tools.list_ports
import numpy as np
import time
import os
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
                             QHeaderView, QPushButton, QComboBox, QCheckBox, QFileDialog,
                             QMessageBox, QGroupBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QRectF
from PyQt5.QtGui import QPainter, QColor, QFont

# 引入滤波库
from scipy.ndimage import median_filter, gaussian_filter

# 配置参数
MATRIX_DIM = 32

# --- QSS 样式表：美化 UI 外观 ---
STYLE_SHEET = """
QMainWindow {
    background-color: #f0f2f5;
}
QGroupBox {
    font-weight: bold;
    border: 2px solid #dce1e6;
    border-radius: 8px;
    margin-top: 15px;
    background-color: white;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #333;
}
QPushButton {
    background-color: #0078d4;
    color: white;
    border-radius: 5px;
    padding: 10px;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #2b88d8;
}
QPushButton#StopBtn {
    background-color: #d83b01;
}
QPushButton#StopBtn:hover {
    background-color: #ef4808;
}
QPushButton#CalibBtn {
    background-color: #5c2d91;
}
QLabel#StatusLabel {
    color: #0078d4;
    font-weight: bold;
    padding: 5px;
    border: 1px solid #c7e0f4;
    background-color: #eff6fc;
    border-radius: 4px;
}
"""


class SerialThread(QThread):
    data_received = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.port = ''
        self.baudrate = 460800
        self.running = False
        self.ser = None

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.ser.reset_input_buffer()
            self.ser.write(b'1')  # 发送开始命令
            self.running = True

            while self.running:
                # 帧头 0xAA 0x55 (2字节) + 数据 (2048字节)
                if self.ser.in_waiting >= 2050:
                    header = self.ser.read(2)
                    if header == b'\xaa\x55':
                        raw_data = self.ser.read(2048)
                        data = np.frombuffer(raw_data, dtype=np.uint16)
                        if data.size == 1024:
                            self.data_received.emit(data.reshape((MATRIX_DIM, MATRIX_DIM)).copy())
                    else:
                        self.ser.read(1)  # 找回同步
                else:
                    time.sleep(0.005)

            self.ser.write(b'2')  # 发送停止命令
            self.ser.close()
        except Exception as e:
            self.error_occurred.emit(str(e))
            self.running = False

    def stop(self):
        self.running = False


class HeatmapWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.matrix = np.zeros((MATRIX_DIM, MATRIX_DIM))
        self.setMinimumSize(480, 480)

    def update_matrix(self, matrix):
        self.matrix = matrix
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        width, height = self.width(), self.height()
        cell_w, cell_h = width / MATRIX_DIM, height / MATRIX_DIM

        # 映射范围设置
        v_min, v_max = 30, 800
        v_range = v_max - v_min

        for r in range(MATRIX_DIM):
            for c in range(MATRIX_DIM):
                val = self.matrix[r, c]
                norm = max(0.0, min(1.0, (val - v_min) / v_range))
                color = QColor()
                color.setHsv(int((1.0 - norm) * 240), 255, 255)
                painter.fillRect(QRectF(c * cell_w, r * cell_h, cell_w + 0.5, cell_h + 0.5), color)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("压力矩阵科研采集系统 v3.0")
        self.resize(1500, 900)
        self.setStyleSheet(STYLE_SHEET)

        # 数据处理参数
        self.base_matrix = np.zeros((MATRIX_DIM, MATRIX_DIM))
        self.is_calibrated = False
        self.save_path = os.path.join(os.getcwd(), "data")
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)

        self.is_recording = False
        self.file_handle = None
        self.setup_ui()

        self.serial_thread = SerialThread()
        self.serial_thread.data_received.connect(self.process_data)
        self.serial_thread.error_occurred.connect(self.handle_error)

        self.count = 0
        self.last_time = time.time()
        self.table_update_count = 0

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(20)

        # 左侧面板
        left_panel = QVBoxLayout()

        # 1. 连接控制组
        conn_group = QGroupBox("📡 硬件连接")
        conn_layout = QVBoxLayout(conn_group)

        cfg_row = QHBoxLayout()
        self.port_combo = QComboBox()
        self.refresh_ports()
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(['115200', '460800', '921600'])
        self.baud_combo.setCurrentText('460800')
        cfg_row.addWidget(QLabel("端口:"))
        cfg_row.addWidget(self.port_combo)
        cfg_row.addWidget(QLabel("波特率:"))
        cfg_row.addWidget(self.baud_combo)
        conn_layout.addLayout(cfg_row)

        ctrl_row = QHBoxLayout()
        self.start_btn = QPushButton("▶ 开始采集")
        self.stop_btn = QPushButton("■ 停止采集")
        self.stop_btn.setObjectName("StopBtn")
        self.calib_btn = QPushButton("🔄 零点校准")
        self.calib_btn.setObjectName("CalibBtn")

        self.start_btn.clicked.connect(self.start_serial)
        self.stop_btn.clicked.connect(self.stop_serial)
        self.calib_btn.clicked.connect(self.reset_calibration)
        self.stop_btn.setEnabled(False)

        ctrl_row.addWidget(self.start_btn)
        ctrl_row.addWidget(self.stop_btn)
        ctrl_row.addWidget(self.calib_btn)
        conn_layout.addLayout(ctrl_row)
        left_panel.addWidget(conn_group)

        # 2. 数据保存组
        save_group = QGroupBox("💾 数据记录")
        save_layout = QVBoxLayout(save_group)
        self.save_cb = QCheckBox("启用 TXT 自动保存 (年月日_时分秒)")
        self.path_btn = QPushButton("📁 更改保存文件夹")
        self.path_btn.setStyleSheet("background-color: #666;")
        self.path_btn.clicked.connect(self.choose_path)

        path_row = QHBoxLayout()
        path_row.addWidget(self.save_cb)
        path_row.addWidget(self.path_btn)
        save_layout.addLayout(path_row)
        self.path_label = QLabel(f"保存路径: {self.save_path}")
        self.path_label.setStyleSheet("color: #777; font-size: 10px;")
        save_layout.addWidget(self.path_label)
        left_panel.addWidget(save_group)

        # 3. 实时状态
        self.status_label = QLabel("FPS: 0 | Max: 0 | Min: 0")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(self.status_label)

        # 4. 热力图
        self.heatmap = HeatmapWidget()
        left_panel.addWidget(self.heatmap)
        layout.addLayout(left_panel, 3)

        # 右侧表格组
        table_group = QGroupBox("📊 原始数据矩阵 (ADC Counts)")
        table_layout = QVBoxLayout(table_group)
        self.table = QTableWidget(MATRIX_DIM, MATRIX_DIM)
        self.setup_table()
        table_layout.addWidget(self.table)
        layout.addWidget(table_group, 4)

    def setup_table(self):
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setStyleSheet("font-size: 7px; border: none;")
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        for r in range(MATRIX_DIM):
            for c in range(MATRIX_DIM):
                self.table.setItem(r, c, QTableWidgetItem("0"))

    def refresh_ports(self):
        self.port_combo.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo.addItems(ports)

    def choose_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择保存文件夹", self.save_path)
        if path:
            self.save_path = path
            self.path_label.setText(f"保存路径: {self.save_path}")

    def reset_calibration(self):
        self.is_calibrated = False
        self.status_label.setText("正在执行零点校准...")

    def start_serial(self):
        if self.save_cb.isChecked():
            try:
                filename = datetime.now().strftime("%Y%m%d_%H%M%S.txt")
                self.file_handle = open(os.path.join(self.save_path, filename), 'w', encoding='utf-8')
                self.is_recording = True
                print(f"创建文件: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "文件错误", str(e))
                return

        port = self.port_combo.currentText()
        if not port: return
        self.serial_thread.port = port
        self.serial_thread.baudrate = int(self.baud_combo.currentText())
        self.serial_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_serial(self):
        self.serial_thread.stop()
        if self.file_handle:
            self.file_handle.flush()
            self.file_handle.close()
            self.file_handle = None
        self.is_recording = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("状态: 已停止")

    def process_data(self, matrix):
        # 0. 零点校准逻辑
        if not self.is_calibrated:
            self.base_matrix = matrix.copy()
            self.is_calibrated = True
            print("校准基准已建立")
            return

        # 1. 去除底噪与死区过滤
        processed = np.maximum(0, matrix.astype(np.int32) - self.base_matrix.astype(np.int32))
        processed[processed < 30] = 0  # 30 为死区阈值

        # 2. 异步保存数据 (保存处理后的净值)
        if self.is_recording and self.file_handle:
            t = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.file_handle.write(f"{t} {' '.join(map(str, processed.flatten()))}\n")
            if self.count % 10 == 0: self.file_handle.flush()

        # 3. 视觉处理 (中值滤波 + 高斯平滑)
        clipped = np.clip(processed, 0, 2500)
        f_matrix = median_filter(clipped, size=3)
        f_matrix = gaussian_filter(f_matrix, sigma=0.5)
        self.heatmap.update_matrix(f_matrix)

        # 4. 表格更新 (限流)
        self.table_update_count += 1
        if self.table_update_count >= 10:
            self.table.setUpdatesEnabled(False)
            for r in range(MATRIX_DIM):
                for c in range(MATRIX_DIM):
                    self.table.item(r, c).setText(str(int(processed[r, c])))
            self.table.setUpdatesEnabled(True)
            self.table_update_count = 0

        # 5. 性能统计
        self.count += 1
        now = time.time()
        if now - self.last_time >= 1.0:
            fps = self.count / (now - self.last_time)
            self.status_label.setText(f"FPS: {fps:.1f} | Max: {np.max(processed)} | Min: {np.min(processed)}")
            self.count, self.last_time = 0, now

    def handle_error(self, msg):
        self.stop_serial()
        QMessageBox.critical(self, "串口异常", msg)

    def closeEvent(self, event):
        self.stop_serial()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())