import sys
import serial
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
import time


class PressureMap(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()

        # 1. 串口配置
        try:
            self.ser = serial.Serial('COM4', 460800, timeout=0.01)
            self.ser.set_buffer_size(rx_size=1024 * 100)
            print("📡 串口已连接，正在同步行数据并校准底噪...")
        except Exception as e:
            print(f"❌ 串口打开失败: {e}")
            sys.exit()

        self.buffer = bytearray()
        # 【关键修改】协议长度：AA BB(2) + RowIdx(1) + Data(64) = 67字节
        self.packet_len = 67

        # --- 处理参数 ---
        self.alpha = 0.2  # 滤波系数
        self.full_matrix = np.zeros((32, 32), dtype=float)
        self.last_matrix = np.zeros((32, 32), dtype=float)

        # --- 底噪校准 ---
        self.base_matrix = np.zeros((32, 32), dtype=float)
        self.calibrating = True
        self.calib_frames = 0
        self.max_calib_frames = 40  # 采样40行包作为初始基准

        # --- FPS 统计 ---
        self.last_time = time.time()
        self.frame_count = 0

        self.init_ui()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1)

    def init_ui(self):
        self.view = self.addViewBox()
        self.view.setAspectLocked(True)
        self.img = pg.ImageItem()
        self.view.addItem(self.img)

        self.view.setRange(QtCore.QRectF(0, 0, 32, 32))
        self.img.setLevels([0, 500])  # 减去底噪后，灵敏度区间调小

        colormap = pg.colormap.get('inferno')
        self.bar = pg.ColorBarItem(values=(0, 500), colorMap=colormap)
        self.bar.setImageItem(self.img)

        self.info_text = pg.TextItem(text="Waiting for Data...", color='w', anchor=(0, 0))
        self.view.addItem(self.info_text)

        self.setWindowTitle("SCU Pressure System - Row Sync Denoise")
        self.resize(800, 800)

    def update_plot(self):
        if self.ser.in_waiting > 0:
            self.buffer.extend(self.ser.read(self.ser.in_waiting))

            # 寻找行包头 AA BB
            while len(self.buffer) >= self.packet_len:
                idx = self.buffer.find(b'\xAA\xBB')
                if idx == -1:
                    self.buffer = self.buffer[-1:]
                    break
                if idx > 0:
                    self.buffer = self.buffer[idx:]
                    continue

                if len(self.buffer) >= self.packet_len:
                    # 1. 提取行号和原始数据
                    row_idx = self.buffer[2]
                    raw_payload = self.buffer[3:self.packet_len]
                    line_data = np.frombuffer(raw_payload, dtype=np.uint16).astype(float)

                    if row_idx < 32:
                        # 2. 存入当前矩阵
                        self.full_matrix[row_idx, :] = line_data

                        # 3. 校准与去噪处理
                        if self.calibrating:
                            self.base_matrix[row_idx, :] += line_data
                            self.calib_frames += 1
                            if self.calib_frames % 32 == 0:  # 粗略显示进度
                                self.info_text.setText(f"Calibrating...")

                            if self.calib_frames >= (self.max_calib_frames * 32):
                                self.base_matrix /= self.max_calib_frames
                                self.calibrating = False
                                print("✅ 传感器校准完成")
                        else:
                            # 去底噪 -> 负值归零 -> 低通滤波
                            processed = self.full_matrix - self.base_matrix
                            processed[processed < 0] = 0

                            # 画面平滑
                            display_mat = self.alpha * processed + (1 - self.alpha) * self.last_matrix
                            self.last_matrix = display_mat.copy()

                            # 更新画面（.T 转置匹配物理方向）
                            self.img.setImage(display_mat.T, autoLevels=False)

                        # 4. FPS 统计
                        self.frame_count += 1
                        now = time.time()
                        if now - self.last_time >= 1.0:
                            fps = (self.frame_count / 32) / (now - self.last_time)
                            if not self.calibrating:
                                self.info_text.setText(f"FPS: {fps:.1f}")
                            self.frame_count = 0
                            self.last_time = now

                    self.buffer = self.buffer[self.packet_len:]
                else:
                    break


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = PressureMap()
    window.show()
    sys.exit(app.exec())