import sys
import os
import numpy as np
import matplotlib

# --- 关键修改：动态将父目录加入 sys.path 确保跨目录导入正常 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 强制使用更稳定的 Agg 后端进行绘图计算
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QTextEdit, QFileDialog, QLabel)
from PyQt5.QtCore import Qt
from vmdpy import VMD

# 修正导入链路
import data_loader
import preprocess
from algorithms import vmd_MAPE, base


class VmdFprStepByStep(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VMD-FPR 逻辑验证工具 - 双寻优准则版")
        self.resize(1600, 1000)

        # 核心数据成员
        self.fs = 10.0
        self.clean_frames = None
        self.signal_1d = None
        self.all_u_cache = {}
        self.results = {
            "rebound": {"best_k": 2, "mapes": [], "recon": None},
            "fast_drop": {"best_k": 2, "mapes": [], "recon": None}
        }

        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # 控制面板
        control_panel = QVBoxLayout()
        self.btn_1 = QPushButton("步骤 1: 数据加载与清洗")
        self.btn_2 = QPushButton("步骤 2: 左右分区 5x5 ROI 提取")
        self.btn_3 = QPushButton("步骤 3: VMD 寻优 (反弹 vs 快速下降@0.0001)")
        self.btn_4 = QPushButton("步骤 4: 重构呼吸信号波形")
        self.btn_5 = QPushButton("步骤 5: FPR 识别与 BPM 计算")

        for btn in [self.btn_1, self.btn_2, self.btn_3, self.btn_4, self.btn_5]:
            btn.setFixedHeight(50)
            control_panel.addWidget(btn)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setStyleSheet("background-color: #2b2b2b; color: #a9b7c6; font-family: 'Consolas';")
        control_panel.addWidget(QLabel("执行日志:"))
        control_panel.addWidget(self.log_edit)
        layout.addLayout(control_panel, 1)

        # 绘图区域
        self.figure = plt.figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 3)

        self.btn_1.clicked.connect(self.run_step_1)
        self.btn_2.clicked.connect(self.run_step_2)
        self.btn_3.clicked.connect(self.run_step_3)
        self.btn_4.clicked.connect(self.run_step_4)
        self.btn_5.clicked.connect(self.run_step_5)

    def log(self, msg):
        self.log_edit.append(f"<b>[INFO]</b> {msg}")
        QApplication.processEvents()

    def run_step_1(self):
        """步骤 1: 加载与清洗"""
        path, _ = QFileDialog.getOpenFileName(self, "选择数据", "", "Text Files (*.txt)")
        if not path: return
        try:
            t, f = data_loader.load_pressure_txt(path)
            _, self.clean_frames = preprocess.clean_dataset(t, f)
            self.log(f"数据加载成功: {len(self.clean_frames)} 帧")
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.imshow(self.clean_frames[len(self.clean_frames) // 2], cmap='jet')
            ax.set_title("预处理后的中间帧热力图")
            self.canvas.draw()
        except Exception as e:
            self.log(f"加载出错: {str(e)}")

    def run_step_2(self):
        """步骤 2: 左右分区 ROI 提取"""
        if self.clean_frames is None: return
        self.signal_1d = base.get_dual_roi_mean(self.clean_frames, window_size=5)
        self.log("5x5 ROI 信号提取完成 (已含小波去噪)")
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.signal_1d, color='#0078d7')
        ax.set_title("1D ROI 均值趋势信号")
        self.canvas.draw()

    def run_step_3(self):
        """步骤 3: VMD 迭代寻优 (区分反弹法与阈值快降法)"""
        if self.signal_1d is None: return
        self.log("执行 VMD 寻优计算 (K=2~10)...")

        mapes = []
        self.all_u_cache = {}
        k_range = list(range(2, 11))
        sig = self.signal_1d

        for k in k_range:
            u, _, _ = VMD(sig, alpha=2000, tau=0, K=k, DC=0, init=1, tol=1e-7)
            res = sig - np.sum(u, axis=0)
            mape = np.sum(res ** 2) / np.sum(sig ** 2)
            mapes.append(mape)
            self.all_u_cache[k] = u
            self.log(f"K={k} | MAPE: {mape:.8f}")

        # --- 寻优算法 A: 触底反弹法 ---
        best_k_reb = 2
        for i in range(1, len(mapes)):
            if mapes[i] > mapes[i - 1]:  # 发现拐点
                best_k_reb = k_range[i - 1]
                break
            best_k_reb = k_range[i]

        # --- 寻优算法 B: 快速下降法 ---
        best_k_fast = 2
        qualified_indices = [i for i, m in enumerate(mapes) if m < 0.0001]

        if not qualified_indices:
            self.log("警告: 所有 K 值的 MAPE 均未达到 0.0001 约束，取最小 MAPE 点")
            best_k_fast = k_range[mapes.index(min(mapes))]
        else:
            diffs = np.abs(np.diff(mapes))
            valid_diff_indices = [i for i in range(len(diffs)) if i + 1 in qualified_indices]
            if valid_diff_indices:
                max_diff_idx = valid_diff_indices[np.argmax([diffs[i] for i in valid_diff_indices])]
                best_k_fast = k_range[max_diff_idx + 1]
            else:
                best_k_fast = k_range[qualified_indices[0]]

        self.results["rebound"]["best_k"] = best_k_reb
        self.results["fast_drop"]["best_k"] = best_k_fast

        # 绘图显示
        self.figure.clear()
        ax_mape = self.figure.add_subplot(111)
        ax_mape.plot(k_range, mapes, 'o-', color='#333333', markerfacecolor='white', markersize=6)
        ax_mape.axhline(y=0.0001, color='orange', linestyle=':', label='快降法门限 (0.0001)')
        ax_mape.scatter(best_k_reb, mapes[k_range.index(best_k_reb)], color='red', s=100,
                        label=f'反弹法 K={best_k_reb}', zorder=5)
        ax_mape.scatter(best_k_fast, mapes[k_range.index(best_k_fast)], color='green', marker='s', s=100,
                        label=f'快降法 K={best_k_fast}', zorder=5)

        ax_mape.set_title("VMD 分解能量残差比 (MAPE) 寻优曲线")
        ax_mape.set_xlabel("K 值")
        ax_mape.set_ylabel("MAPE")
        ax_mape.legend()
        self.canvas.draw()
        self.log(f"判定结果: 触底反弹 K={best_k_reb}, 快速下降(约束后) K={best_k_fast}")

    def run_step_4(self):
        """步骤 4: 重构波形"""
        if not self.all_u_cache: return
        self.log("正在执行信号重构...")
        k_r = self.results["rebound"]["best_k"]
        k_f = self.results["fast_drop"]["best_k"]

        self.results["rebound"]["recon"] = vmd_MAPE.reconstruct_respiration_signal(self.all_u_cache[k_r], self.fs)
        self.results["fast_drop"]["recon"] = vmd_MAPE.reconstruct_respiration_signal(self.all_u_cache[k_f], self.fs)

        self.figure.clear()
        ax1 = self.figure.add_subplot(211)
        ax1.plot(self.results["rebound"]["recon"], color='#e74c3c')
        ax1.set_title(f"反弹法重构波形 (K={k_r})")

        ax2 = self.figure.add_subplot(212)
        ax2.plot(self.results["fast_drop"]["recon"], color='#2ecc71')
        ax2.set_title(f"快速下降法重构波形 (K={k_f})")
        self.figure.tight_layout()
        self.canvas.draw()

    def run_step_5(self):
        """步骤 5: FPR 频率计算"""
        if self.results["rebound"]["recon"] is None: return
        self.log("正在执行 FPR 特征识别算法...")

        bpm_r = vmd_MAPE.calculate_bpm_fpr(self.results["rebound"]["recon"], self.fs)
        bpm_f = vmd_MAPE.calculate_bpm_fpr(self.results["fast_drop"]["recon"], self.fs)

        self.figure.axes[0].set_title(f"反弹法 (K={self.results['rebound']['best_k']}) | {bpm_r:.2f} BPM")
        self.figure.axes[1].set_title(f"快速下降法 (K={self.results['fast_drop']['best_k']}) | {bpm_f:.2f} BPM")
        self.canvas.draw()
        self.log(f"最终结果: 反弹法 {bpm_r:.2f} BPM, 快降法 {bpm_f:.2f} BPM")


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    app = QApplication(sys.argv)
    window = VmdFprStepByStep()
    window.show()
    sys.exit(app.exec_())