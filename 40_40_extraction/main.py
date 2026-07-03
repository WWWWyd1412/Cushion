import sys
import os
import ctypes

# 预加载 Qt DLL 到进程内存，使 PyQt5 .pyd 能在任何启动方式下找到它们
_qt_lib = r'D:\anaconda\envs\torch\Library\bin'
for _dll in ['Qt5Core_conda', 'Qt5Gui_conda', 'Qt5Widgets_conda',
             'Qt5Network_conda', 'Qt5PrintSupport_conda']:
    try:
        ctypes.CDLL(os.path.join(_qt_lib, _dll + '.dll'))
    except OSError:
        pass

import serial
import serial.tools.list_ports
import numpy as np
import time
import os
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QComboBox,
                             QCheckBox, QFileDialog, QMessageBox, QGroupBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QRectF
from PyQt5.QtGui import QPainter, QColor, QFont

MATRIX_DIM = 40
FRAME_HEADER = bytes([0x5A, 0x01, 0x95, 0x6C])
FRAME_SIZE = 4 + 2 + 3200  # header + src_addr + data

STYLE_SHEET = """
QMainWindow { background-color: #f0f2f5; }
QGroupBox {
    font-weight: bold; border: 2px solid #dce1e6;
    border-radius: 8px; margin-top: 15px; background-color: white;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #333; }
QPushButton {
    background-color: #0078d4; color: white;
    border-radius: 5px; padding: 10px; font-size: 13px;
}
QPushButton:hover { background-color: #2b88d8; }
QPushButton#StopBtn { background-color: #d83b01; }
QPushButton#StopBtn:hover { background-color: #ef4808; }
QLabel#StatusLabel {
    color: #0078d4; font-weight: bold; padding: 5px;
    border: 1px solid #c7e0f4; background-color: #eff6fc; border-radius: 4px;
}
"""


class SerialThread(QThread):
    data_received = pyqtSignal(np.ndarray, int)  # matrix, source_addr
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.port = ''
        self.baudrate = 460800
        self.running = False
        self.ser = None
        self._buf = bytearray()

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.ser.reset_input_buffer()
            self.running = True

            while self.running:
                chunk = self.ser.read(self.ser.in_waiting or 1)
                if chunk:
                    self._buf.extend(chunk)
                    self._parse()
                else:
                    time.sleep(0.005)

            self.ser.close()
        except Exception as e:
            self.error_occurred.emit(str(e))
            self.running = False

    def _parse(self):
        while len(self._buf) >= FRAME_SIZE:
            idx = self._buf.find(FRAME_HEADER)
            if idx == -1:
                self._buf = self._buf[-3:]  # keep potential partial header
                return
            if idx > 0:
                del self._buf[:idx]
            if len(self._buf) < FRAME_SIZE:
                return

            frame = self._buf[:FRAME_SIZE]
            del self._buf[:FRAME_SIZE]

            src_addr = int.from_bytes(frame[4:6], 'big')
            raw = np.frombuffer(frame[6:], dtype='<u2')  # little-endian uint16
            if raw.size == MATRIX_DIM * MATRIX_DIM:
                self.data_received.emit(raw.reshape(MATRIX_DIM, MATRIX_DIM).copy(), src_addr)

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
        painter.setPen(Qt.NoPen)
        cw, ch = self.width() / MATRIX_DIM, self.height() / MATRIX_DIM
        # 动态范围：用5%~95%分位数避免噪点撑满量程
        v_min = float(np.percentile(self.matrix, 5))
        v_max = float(np.percentile(self.matrix, 95))
        v_range = max(v_max - v_min, 1.0)
        color = QColor()
        for r in range(MATRIX_DIM):
            for c in range(MATRIX_DIM):
                norm = max(0.0, min(1.0, (self.matrix[r, c] - v_min) / v_range))
                # 传感器反向：值越低压力越大 → norm小=红(0°), norm大=蓝(240°)
                color.setHsv(int(norm * 240), 255, 255)
                painter.fillRect(QRectF(c * cw, r * ch, cw + 0.5, ch + 0.5), color)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("40×40 压力矩阵实时采集 v1.0")
        self.resize(700, 700)
        self.setStyleSheet(STYLE_SHEET)

        self.save_path = os.path.join(os.getcwd(), "data")
        os.makedirs(self.save_path, exist_ok=True)
        self.file_handle = None
        self.is_recording = False
        self.count = 0
        self.last_time = time.time()

        self._build_ui()

        self.thread = SerialThread()
        self.thread.data_received.connect(self._on_data)
        self.thread.error_occurred.connect(self._on_error)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # --- 连接控制 ---
        conn_group = QGroupBox("📡 硬件连接")
        conn_lay = QVBoxLayout(conn_group)
        row1 = QHBoxLayout()
        self.port_combo = QComboBox()
        self._refresh_ports()
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(['115200', '460800', '921600'])
        self.baud_combo.setCurrentText('460800')
        row1.addWidget(QLabel("端口:"))
        row1.addWidget(self.port_combo)
        row1.addWidget(QLabel("波特率:"))
        row1.addWidget(self.baud_combo)
        conn_lay.addLayout(row1)

        row2 = QHBoxLayout()
        self.start_btn = QPushButton("▶ 开始采集")
        self.stop_btn = QPushButton("■ 停止")
        self.stop_btn.setObjectName("StopBtn")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        row2.addWidget(self.start_btn)
        row2.addWidget(self.stop_btn)
        conn_lay.addLayout(row2)
        layout.addWidget(conn_group)

        # --- 数据保存 ---
        save_group = QGroupBox("💾 数据记录")
        save_lay = QHBoxLayout(save_group)
        self.save_cb = QCheckBox("启用 TXT 自动保存")
        path_btn = QPushButton("📁 保存路径")
        path_btn.setStyleSheet("background-color:#666;")
        path_btn.clicked.connect(self._choose_path)
        self.path_label = QLabel(f"路径: {self.save_path}")
        self.path_label.setStyleSheet("color:#777; font-size:10px;")
        save_lay.addWidget(self.save_cb)
        save_lay.addWidget(path_btn)
        save_lay.addWidget(self.path_label, 1)
        layout.addWidget(save_group)

        # --- 状态栏 ---
        self.status_label = QLabel("FPS: 0 | Max: 0 | Min: 0 | 源地址: -")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # --- 热力图 ---
        self.heatmap = HeatmapWidget()
        layout.addWidget(self.heatmap, 1)

    def _refresh_ports(self):
        self.port_combo.clear()
        self.port_combo.addItems([p.device for p in serial.tools.list_ports.comports()])

    def _choose_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择保存文件夹", self.save_path)
        if path:
            self.save_path = path
            self.path_label.setText(f"路径: {self.save_path}")

    def _start(self):
        if self.save_cb.isChecked():
            try:
                fname = datetime.now().strftime("%Y%m%d_%H%M%S.txt")
                self.file_handle = open(os.path.join(self.save_path, fname), 'w', encoding='utf-8')
                self.is_recording = True
            except Exception as e:
                QMessageBox.critical(self, "文件错误", str(e))
                return

        port = self.port_combo.currentText()
        if not port:
            return
        self.thread.port = port
        self.thread.baudrate = int(self.baud_combo.currentText())
        self.thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _stop(self):
        self.thread.stop()
        if self.file_handle:
            self.file_handle.flush()
            self.file_handle.close()
            self.file_handle = None
        self.is_recording = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("状态: 已停止")

    def _on_data(self, matrix: np.ndarray, src_addr: int):
        if self.is_recording and self.file_handle:
            t = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.file_handle.write(f"{t} {' '.join(map(str, matrix.flatten()))}\n")
            if self.count % 10 == 0:
                self.file_handle.flush()

        self.heatmap.update_matrix(matrix)

        self.count += 1
        now = time.time()
        if now - self.last_time >= 1.0:
            fps = self.count / (now - self.last_time)
            self.status_label.setText(
                f"FPS: {fps:.1f} | Max: {matrix.max()} | Min: {matrix.min()} | 源地址: {src_addr}"
            )
            self.count, self.last_time = 0, now

    def _on_error(self, msg: str):
        self._stop()
        QMessageBox.critical(self, "串口异常", msg)

    def closeEvent(self, event):
        self._stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
