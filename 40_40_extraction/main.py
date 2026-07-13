# -*- coding: utf-8 -*-
"""
40×40 压力矩阵实时采集 v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
功能:
  - 串口读取40×40压力矩阵数据
  - 实时热力图显示
  - 设置采集时长（秒），倒计时自动结束
  - 自动保存为 YYYYMMDD_HHMMSS_40x40.txt
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys
import os
import ctypes
import time
from datetime import datetime

# 预加载 Qt DLL
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

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QGroupBox,
    QFileDialog, QMessageBox
)
from PyQt5.QtCore  import QThread, pyqtSignal, Qt, QRectF, QTimer
from PyQt5.QtGui   import QPainter, QColor, QFont

# ── 常量 ────────────────────────────────────────────────────────
MATRIX_DIM  = 40
FRAME_HEADER = bytes([0x5A, 0x01, 0x95, 0x6C])
FRAME_SIZE   = 4 + 2 + 3200   # header(4) + src_addr(2) + data(3200)

STYLE_SHEET = """
QMainWindow { background-color: #f0f2f5; }
QGroupBox {
    font-weight: bold; border: 2px solid #dce1e6;
    border-radius: 8px; margin-top: 15px; background-color: white;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #333; }
QPushButton {
    background-color: #0078d4; color: white;
    border-radius: 5px; padding: 8px 16px; font-size: 13px;
}
QPushButton:hover  { background-color: #2b88d8; }
QPushButton:disabled { background-color: #aaa; }
QPushButton#StopBtn { background-color: #d83b01; }
QPushButton#StopBtn:hover { background-color: #ef4808; }
QLabel#StatusLabel {
    color: #0078d4; font-weight: bold; padding: 6px;
    border: 1px solid #c7e0f4; background-color: #eff6fc;
    border-radius: 4px; font-size: 13px;
}
QLabel#CountdownLabel {
    color: #d83b01; font-weight: bold; font-size: 20px;
    padding: 4px 12px; border: 2px solid #d83b01;
    border-radius: 6px; background-color: #fff4f0;
}
"""


# ════════════════════════════════════════════════════════════════
# 串口读取线程
# ════════════════════════════════════════════════════════════════
class SerialThread(QThread):
    data_received = pyqtSignal(np.ndarray, int)   # (matrix, src_addr)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.port     = ''
        self.baudrate = 460800
        self.running  = False
        self.ser      = None
        self._buf     = bytearray()

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
                self._buf = self._buf[-3:]
                return
            if idx > 0:
                del self._buf[:idx]
            if len(self._buf) < FRAME_SIZE:
                return
            frame = self._buf[:FRAME_SIZE]
            del self._buf[:FRAME_SIZE]
            src_addr = int.from_bytes(frame[4:6], 'big')
            raw = np.frombuffer(frame[6:], dtype='<u2')
            if raw.size == MATRIX_DIM * MATRIX_DIM:
                self.data_received.emit(
                    raw.reshape(MATRIX_DIM, MATRIX_DIM).copy(), src_addr
                )

    def stop(self):
        self.running = False


# ════════════════════════════════════════════════════════════════
# 热力图控件
# ════════════════════════════════════════════════════════════════
class HeatmapWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.matrix = np.zeros((MATRIX_DIM, MATRIX_DIM))
        self.setMinimumSize(480, 480)

    def update_matrix(self, matrix: np.ndarray):
        self.matrix = matrix
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(Qt.NoPen)
        cw = self.width()  / MATRIX_DIM
        ch = self.height() / MATRIX_DIM

        # 5%~95% 分位数动态范围，避免噪点撑满量程
        v_min = float(np.percentile(self.matrix, 5))
        v_max = float(np.percentile(self.matrix, 95))
        v_range = max(v_max - v_min, 1.0)

        color = QColor()
        for r in range(MATRIX_DIM):
            for c in range(MATRIX_DIM):
                norm = max(0.0, min(1.0, (self.matrix[r, c] - v_min) / v_range))
                # norm小=红(0°)压力大, norm大=蓝(240°)压力小
                color.setHsv(int(norm * 240), 255, 255)
                painter.fillRect(QRectF(c*cw, r*ch, cw+0.5, ch+0.5), color)


# ════════════════════════════════════════════════════════════════
# 主窗口
# ════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("40×40 压力矩阵采集 v2.0")
        self.resize(720, 760)
        self.setStyleSheet(STYLE_SHEET)

        # 保存路径
        self.save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '40_40_Cushion_Data'
        )
        os.makedirs(self.save_path, exist_ok=True)

        self.file_handle   = None
        self.is_recording  = False
        self.frame_count   = 0
        self.fps_count     = 0j
        self.last_fps_time = time.time()
        self._remain_sec   = 0           # 倒计时剩余秒数

        self._build_ui()

        # 串口线程
        self.thread = SerialThread()
        self.thread.data_received.connect(self._on_data)
        self.thread.error_occurred.connect(self._on_error)

        # 倒计时定时器（每秒触发）
        self.countdown_timer = QTimer(self)
        self.countdown_timer.setInterval(1000)
        self.countdown_timer.timeout.connect(self._on_tick)

    # ── UI 构建 ──────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ── 串口配置 ──
        conn_grp = QGroupBox("📡 串口连接")
        conn_lay = QHBoxLayout(conn_grp)

        self.port_combo = QComboBox()
        self._refresh_ports()
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(['115200', '460800', '921600'])
        self.baud_combo.setCurrentText('460800')
        refresh_btn = QPushButton("🔄")
        refresh_btn.setFixedWidth(36)
        refresh_btn.setToolTip("刷新端口")
        refresh_btn.clicked.connect(self._refresh_ports)

        conn_lay.addWidget(QLabel("端口:"))
        conn_lay.addWidget(self.port_combo, 1)
        conn_lay.addWidget(refresh_btn)
        conn_lay.addSpacing(12)
        conn_lay.addWidget(QLabel("波特率:"))
        conn_lay.addWidget(self.baud_combo)
        layout.addWidget(conn_grp)

        # ── 采集控制 ──
        ctrl_grp = QGroupBox("⏱ 采集控制")
        ctrl_lay = QHBoxLayout(ctrl_grp)

        ctrl_lay.addWidget(QLabel("采集时长(秒):"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(5, 3600)
        self.duration_spin.setValue(120)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setFixedWidth(90)
        ctrl_lay.addWidget(self.duration_spin)

        ctrl_lay.addSpacing(20)
        self.start_btn = QPushButton("▶ 开始采集")
        self.stop_btn  = QPushButton("■ 停止")
        self.stop_btn.setObjectName("StopBtn")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        ctrl_lay.addWidget(self.start_btn)
        ctrl_lay.addWidget(self.stop_btn)

        ctrl_lay.addSpacing(20)
        ctrl_lay.addWidget(QLabel("保存路径:"))
        path_btn = QPushButton("📁")
        path_btn.setFixedWidth(36)
        path_btn.clicked.connect(self._choose_path)
        ctrl_lay.addWidget(path_btn)
        self.path_label = QLabel(os.path.abspath(self.save_path))
        self.path_label.setStyleSheet("color:#777; font-size:10px;")
        ctrl_lay.addWidget(self.path_label, 1)
        layout.addWidget(ctrl_grp)

        # ── 文件名 ──
        from PyQt5.QtWidgets import QLineEdit
        name_grp = QGroupBox("📄 文件命名")
        name_lay = QHBoxLayout(name_grp)
        name_lay.addWidget(QLabel("文件名:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("留空则自动生成  YYYYMMDD_HHMMSS_40x40.txt")
        self.name_edit.setToolTip("可选。不含路径，扩展名可省略（自动补 .txt）")
        name_lay.addWidget(self.name_edit, 1)
        layout.addWidget(name_grp)

        # ── 状态行 ──
        stat_row = QHBoxLayout()
        self.status_label = QLabel("状态: 就绪")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.countdown_label = QLabel("")
        self.countdown_label.setObjectName("CountdownLabel")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setFixedWidth(100)
        stat_row.addWidget(self.status_label, 1)
        stat_row.addWidget(self.countdown_label)
        layout.addLayout(stat_row)

        # ── 热力图 ──
        self.heatmap = HeatmapWidget()
        layout.addWidget(self.heatmap, 1)

    # ── 辅助方法 ─────────────────────────────────────────────────
    def _refresh_ports(self):
        self.port_combo.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo.addItems(ports if ports else ["(无可用端口)"])

    def _choose_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择保存文件夹",
                                                self.save_path)
        if path:
            self.save_path = path
            self.path_label.setText(path)

    # ── 采集控制 ─────────────────────────────────────────────────
    def _start(self):
        port = self.port_combo.currentText()
        if not port or port.startswith("("):
            QMessageBox.warning(self, "提示", "请先选择串口端口")
            return

        # 打开保存文件（优先使用用户输入的文件名）
        try:
            custom = self.name_edit.text().strip()
            if custom:
                # 确保以 .txt 结尾
                if not custom.lower().endswith('.txt'):
                    custom += '.txt'
                fname = custom
            else:
                fname = datetime.now().strftime("%Y%m%d_%H%M%S") + "_40x40.txt"
            fpath = os.path.join(self.save_path, fname)
            self.file_handle  = open(fpath, 'w', encoding='utf-8')
            self.is_recording = True
        except Exception as e:
            QMessageBox.critical(self, "文件错误", str(e))
            return

        # 启动串口线程
        self.thread.port     = port
        self.thread.baudrate = int(self.baud_combo.currentText())
        self.thread.start()

        # 启动倒计时
        self._remain_sec = self.duration_spin.value()
        self._update_countdown_label()
        self.countdown_timer.start()

        self.frame_count = self.fps_count = 0
        self.last_fps_time = time.time()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.duration_spin.setEnabled(False)
        self.status_label.setText(
            f"采集中 → 保存至: {os.path.basename(fpath)}"
        )

    def _stop(self):
        # 停止倒计时和串口线程
        self.countdown_timer.stop()
        self.thread.stop()
        self.thread.wait(2000)

        # 关闭文件
        if self.file_handle:
            self.file_handle.flush()
            self.file_handle.close()
            self.file_handle = None

        self.is_recording = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.duration_spin.setEnabled(True)
        self.countdown_label.setText("")
        self.status_label.setText(
            f"已停止 — 共采集 {self.frame_count} 帧"
        )

    def _on_tick(self):
        """每秒触发一次：倒计时递减，归零时自动停止"""
        self._remain_sec -= 1
        self._update_countdown_label()
        if self._remain_sec <= 0:
            self._stop()

    def _update_countdown_label(self):
        m, s = divmod(self._remain_sec, 60)
        self.countdown_label.setText(f"⏳ {m:02d}:{s:02d}")

    # ── 数据回调 ─────────────────────────────────────────────────
    def _on_data(self, matrix: np.ndarray, src_addr: int):
        # 传感器ADC输出：高值=无压力，低值=有压力 → 反转为直觉方向（高=压力大）
        pressure = (4095 - matrix).clip(0)

        # 写入文件: 时间戳 + 1600个整数（已反转的压力值）
        if self.is_recording and self.file_handle:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.file_handle.write(
                f"{ts} {' '.join(map(str, pressure.flatten()))}\n"
            )
            self.frame_count += 1
            if self.frame_count % 50 == 0:
                self.file_handle.flush()

        # 更新热力图（用反转后的压力值）
        self.heatmap.update_matrix(pressure)

        # FPS 统计
        self.fps_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            fps = self.fps_count / (now - self.last_fps_time)
            self.status_label.setText(
                f"采集中  FPS: {fps:.1f} | Max: {matrix.max()} "
                f"| Min: {matrix.min()} | 源地址: {src_addr:#06x} "
                f"| 已采集: {self.frame_count} 帧"
            )
            self.fps_count = 0
            self.last_fps_time = now

    def _on_error(self, msg: str):
        self._stop()
        QMessageBox.critical(self, "串口异常", msg)

    def closeEvent(self, event):
        self._stop()
        super().closeEvent(event)


# ════════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
