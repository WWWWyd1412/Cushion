---
name: new-method
description: 为新 CUSHION 项目生成新算法方法、测试脚本或新分析模块。当用户说"添加新算法"、"创建新方法"、"写个测试脚本"、"新建一个分析模块"时触发。覆盖在 Breath_Extraction/algorithms/、HeartbeatRate/algorithms/ 等位置添加代码，以及在 algorithms/test_scripts/ 创建分步验证 GUI。
---

# 新 CUSHION 项目代码生成技能

这个技能帮助你按照项目既有规范和模式，在 new_CUSHION 项目中生成新的算法方法、测试脚本或全新的分析模块。

## 项目架构总览

```
new_CUSHION/
├── Breath_Extraction/          # 离线呼吸提取 (PyQt5 滑窗分析 GUI)
│   ├── algorithms/
│   │   ├── base.py             # 共享工具函数 (243行)
│   │   ├── __init__.py         # 算法注册表
│   │   ├── emd_extract.py      # EMD 分解
│   │   ├── vmd_extract.py      # VMD 分解
│   │   ├── vmd_MAPE.py         # VMD + MAPE 参数寻优
│   │   ├── afd_extract.py      # AFD 自适应傅里叶分解
│   │   ├── smvmd_extract.py    # SMVMD 空间多通道变分模态分解
│   │   ├── goa_vmd_extract.py  # GOA-VMD 蚱蜢优化 VMD
│   │   └── test_scripts/       # 分步验证 GUI 工具
│   ├── main_UI.py              # ★滑窗分析主界面 (深色主题, 3子图, ←→键导航)
│   ├── data_loader.py          # TXT 压力矩阵解析
│   └── preprocess.py           # 数据清洗预处理
├── HeartbeatRate/              # 心跳节律提取 (新增模块)
│   ├── algorithms/
│   │   ├── base.py             # 心跳专用工具 (343行, 含 VME)
│   │   ├── __init__.py
│   │   ├── acmd_extract.py     # ACMD 自适应啁啾模型分解
│   │   ├── emd_extract.py      # EMD
│   │   ├── vmd_extract.py      # VMD (K=6)
│   │   └── vme_extract.py      # VME 变分模态提取
│   ├── main_UI.py              # ★滑窗分析主界面 (深色主题, 3子图, ←→键导航)
│   ├── data_loader.py          # 同 Breath_Extraction
│   ├── preprocess.py           # 同 Breath_Extraction
│   └── test_heartbeat.py       # 控制台测试脚本
├── Real_Time_Extraction/       # 实时采集 (串口)
│   ├── algorithms/
│   │   └── base.py             # 实时版简化工具 (61行)
│   ├── main_RealTime.py        # 实时 GUI (QThread + pyqtgraph)
│   └── ...
├── Test_4_4/                   # 4x4 小垫子串口调试
├── data/                       # TXT 压力矩阵数据
└── papers/                     # 文献综述
```

## 核心约定

### 命名规范
- 算法文件: `{method}_extract.py` (如 `emd_extract.py`, `vmd_extract.py`)
- 入口函数: `extract_respiration(frames, fs)` (呼吸) 或 `extract_heartbeat(frames, fs)` (心跳)
- 测试脚本: `{method}_fpr_standalone.py` (分步验证 GUI) 或 `test_{feature}.py` (控制台)
- 辅助函数使用 `snake_case`

### 数据流
```
TXT文件 → data_loader.load_pressure_txt() → (timestamps, frames[N,32,32])
→ preprocess.clean_dataset() → 去噪帧
→ algorithms/xxx_extract.extract_xxx() → 1D 生理信号
→ calculate_bpm() / calculate_bpm_fpr() → BPM
```

### 导入模式
```python
# 算法模块内的相对导入
from .base import get_dual_roi_mean, butter_bandpass_filter, ...

# 测试脚本的跨目录导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 所有 UI 文件开头的 Conda DLL 路径修复
_conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)
_qt_dll_path = os.path.join(_conda_prefix, 'Library', 'bin')
if os.path.isdir(_qt_dll_path) and _qt_dll_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _qt_dll_path + os.pathsep + os.environ.get('PATH', '')

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
```

---

## 任务一：在现有模块中添加新算法

当用户在 Breath_Extraction、HeartbeatRate 或 Real_Time_Extraction 中添加新的分解/提取算法时，按以下步骤操作：

### 第一步：确定目标模块和信号类型

| 模块 | 信号类型 | 带通范围 | VMD K | 输出函数名 |
|------|----------|----------|-------|-----------|
| Breath_Extraction | 呼吸 | [0.1, 0.5] Hz | 5 | `extract_respiration(frames, fs)` |
| HeartbeatRate | 心跳 | [0.8, 2.2] Hz | 6 | `extract_heartbeat(frames, fs)` |
| Real_Time_Extraction | 呼吸(实时) | 宽松 | 3 | `extract_respiration(frames, fs)` |

### 第二步：在 `algorithms/` 下创建 `{method}_extract.py`

算法文件的**标准骨架**（以呼吸为例）：

```python
import numpy as np
from .base import (
    get_dual_roi_mean,
    reconstruct_multicomponent_with_snr,  # 呼吸用多分量重构
    # select_best_component,              # 心跳用单分量选择
    butter_bandpass_filter,
    wavelet_denoise,
)


def extract_respiration(frames: np.ndarray, fs: float, **kwargs) -> np.ndarray:
    """
    使用 XXX 方法提取呼吸波形

    Parameters
    ----------
    frames : np.ndarray, shape (N, 32, 32)
        压力矩阵序列
    fs : float
        采样频率 (Hz)

    Returns
    -------
    np.ndarray
        提取的 1D 呼吸信号
    """
    # 1. 空间降维 + 滤波去噪
    signal_1d = get_dual_roi_mean(frames)

    # 2. 应用特定分解方法
    # TODO: 在此调用具体的分解算法
    components = your_decomposition_method(signal_1d, ...)

    # 3. 基于频率 + SNR 重构
    # 呼吸: 使用 reconstruct_multicomponent_with_snr()
    result = reconstruct_multicomponent_with_snr(components, fs)
    # 心跳: 使用 select_best_component()
    # result, best_idx, bpm_est = select_best_component(components, fs)

    return result
```

**心跳版本的关键差异：**
- 函数名改为 `extract_heartbeat(frames, fs)`
- 重构使用 `select_best_component(components, fs)` 而非 `reconstruct_multicomponent_with_snr()`
- `get_dual_roi_mean()` 内部已含 VME 基线漂移去除
- 小波去噪参数更温和 (`alpha=0.3` vs `alpha=0.5`)

### 第三步：更新 `algorithms/__init__.py` 注册算法

在 `__init__.py` 中添加导入语句：

```python
# 导入新算法
from .xxx_extract import extract_respiration as extract_xxx

# 在 __all__ 中添加
__all__ = [
    ...
    "extract_xxx",
]
```

注意：`__init__.py` 中的字符串列表**不要漏掉逗号**（已知 bug：Breath_Extraction 的 `__init__.py` 第 24 行缺少逗号导致字符串拼接）。

### 第四步：在 `main_UI.py` 中注册新算法

当前 main_UI.py 使用统一的 `_call_algorithm()` 调度方法。要添加新算法，只需在 `_call_algorithm` 方法中增加一个分支：

```python
def _call_algorithm(self, method, frames, fs):
    """统一调度所有算法 — 在此处添加新算法"""
    if method == "EMD":
        return algorithms.extract_emd(frames, fs)
    elif method == "VMD":
        return algorithms.extract_vmd(frames, fs)
    elif method == "AFD":
        return algorithms.extract_afd(frames, fs)
    elif method == "VMD_FPR":
        return algorithms.extract_vmd_fpr(frames, fs)
    elif method == "GOA-VMD":
        return algorithms.extract_goa_vmd(frames, fs)
    # === 添加新算法分支 ===
    # elif method == "XXX":
    #     return algorithms.extract_xxx(frames, fs)
    else:
        raise ValueError(f"未知算法: {method}")
```

同时在 `setup_ui()` 的 `algo_selector` 中添加算法名称：

```python
self.algo_selector.addItems(["EMD", "VMD", "AFD", "VMD_FPR", "GOA-VMD", "XXX"])
```

### main_UI.py 滑窗架构概览

新版 UI 采用深色主题 + 滑动窗口分析模式:

```
BreathSlidingWindowUI (QMainWindow)
├── 左侧控制面板
│   ├── 1. 数据加载与清洗 (btn_select_file, btn_preprocess)
│   ├── 2. 算法与参数配置 (algo_selector, bpm_method_selector)
│   ├── 3. 滑动窗口设置
│   │   ├── 窗口大小 (帧) — 默认 250
│   │   ├── 窗口步长 (帧) — 默认 50
│   │   ├── 相位自动对齐 (checkbox)
│   │   └── 相对时间轴显示 (checkbox)
│   ├── 4. 执行与查看 (btn_analyze, combo_window_select)
│   └── 状态 + 日志面板
├── 右侧绘图区 (3 子图)
│   ├── [311] BPM 趋势折线图
│   ├── [312] 呼吸波形 (分帧/融合)
│   └── [313] PSD 功率谱密度
└── 键盘导航
    ├── ← 键: 上一窗口
    └── → 键: 下一窗口
```

**核心方法:**
| 方法 | 功能 |
|------|------|
| `run_sliding_analysis(method)` | 按 (win_size, step) 切分帧, 逐窗调用算法, 缓存结果 |
| `plot_sliding_window_results()` | 渲染 3 子图: BPM 趋势 → 波形 → PSD |
| `_call_algorithm(method, frames, fs)` | 统一算法调度入口 |
| `_calc_bpm(signal, fs)` | 统一 BPM 计算 (Peak / FPR) |
| `keyPressEvent(event)` | ← → 键切换滑窗, 自动避开 QLineEdit 焦点 |
| `_mark_psd_peak(ax, f, psd, band)` | PSD 谱峰标注 (呼吸带 0.1-0.5Hz / 心跳带 0.8-2.2Hz) |

**窗口融合:**
- 最后一项 "完整拼接波形" 使用 Tukey-like 加窗重叠相加 (Overlap-Add)
- 渐变窗 taper_ratio=0.15, 确保交界平滑过渡

---

## 任务一点五：创建滑窗分析 UI

当用户需要为算法模块创建完整的滑窗分析 GUI 时，按以下架构创建。

### 滑窗 UI vs 分步验证 UI 的区别

| 特性 | 滑窗 UI (main_UI.py) | 分步验证 UI (test_scripts/) |
|------|---------------------|---------------------------|
| 数据输入 | 一键加载+清洗 | 分步按钮 |
| 分析模式 | 滑窗 + 全局两种 | 单窗 |
| 算法支持 | 全部算法 | 单项算法 |
| 子图布局 | 3 子图 (趋势/波形/PSD) | 通常 1-2 子图 |
| 窗口导航 | 下拉框 + ← → 键 | 无 |
| 融合视图 | 加窗重叠相加拼接 | 无 |
| 主题 | 深色 (#1a1a1e) | 可选深色 |

### 滑窗分析核心流程

```python
def run_sliding_analysis(self, method):
    # 1. 读取滑窗参数
    win_size_frames = int(self.win_size_input.text())   # 默认 250
    step_size_frames = int(self.step_size_input.text())  # 默认 50
    
    # 2. 划分窗口
    windows = []
    start_idx = 0
    while start_idx + win_size_frames <= total_frames:
        windows.append((start_idx, start_idx + win_size_frames))
        start_idx += step_size_frames
    
    # 3. 逐窗处理
    for idx, (start, end) in enumerate(windows):
        window_frames = self.clean_frames[start:end]
        raw_signal = self._call_algorithm(method, window_frames, self.fs)
        smoothed = base.smooth_xxx_signal(raw_signal)
        
        # 相位对齐 (重叠区互相关)
        if idx > 0 and self.cb_align_phase.isChecked():
            overlap_len = win_size_frames - step_size_frames
            corr = np.dot(prev_overlap, curr_overlap)
            if corr < 0:
                smoothed = -smoothed
        
        bpm = self._calc_bpm(smoothed, self.fs)
        
        self.sliding_results.append({
            'window_idx': idx,
            'start_time': start / self.fs,
            'end_time': end / self.fs,
            'center_time': (start + end) / 2 / self.fs,
            'wave': smoothed,
            'bpm': bpm
        })
    
    # 4. 添加融合视图选项
    self.combo_window_select.addItem("完整拼接波形 (全段融合显示)")
```

### 关键参数速查

| 模块 | 窗口大小 | 步长 | PSD 频带 | PSD xlim | 平滑函数 | 寻峰间距 |
|------|---------|------|---------|---------|---------|---------|
| Breath | 250帧(25s) | 50帧(5s) | [0.1,0.5]Hz | [0,1.5]Hz | `smooth_respiration_signal(41)` | `fs*1.2` |
| Heartbeat | 15s | 3s | [0.8,2.2]Hz | [0,4.0]Hz | `smooth_heartbeat_signal(7)` | `fs*0.4` |

---

## 任务二：创建测试脚本

### A. 分步验证 GUI（放在 `algorithms/test_scripts/`）

当用户需要为特定算法创建分步验证工具时，参考以下模板：

**文件命名:** `{method}_fpr_standalone.py` 或 `{method}_standalone.py`

**标准结构（参考 `smvmd_fpr_standalone.py`、`vmd_fpr_standalone.py`、`acmd_fpr_standalone.py`）：**

```python
"""
{Method} + FPR BPM 分步验证工具
"""
import sys, os
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QTextEdit, QLabel, QFileDialog, QGroupBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# 跨目录导入
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_loader import load_pressure_txt
from preprocess import clean_dataset
from algorithms.base import (
    get_dual_roi_mean, calculate_bpm_fpr, calculate_bpm,
    butter_bandpass_filter, calculate_snr, ...
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("{Method} 分步验证")
        self.data = None
        self.fs = 10.0
        self.init_ui()

    def init_ui(self):
        """创建分步按钮 + 状态输出 + matplotlib 画布"""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # 左侧控制面板
        left = QVBoxLayout()

        # 分步按钮 (每步一个 QGroupBox 或 QPushButton)
        btn_step1 = QPushButton("步骤1: 加载数据")
        btn_step1.clicked.connect(self.step1_load)
        left.addWidget(btn_step1)

        btn_step2 = QPushButton("步骤2: ROI 提取")
        btn_step2.clicked.connect(self.step2_roi)
        left.addWidget(btn_step2)

        btn_step3 = QPushButton("步骤3: 算法分解")
        btn_step3.clicked.connect(self.step3_decompose)
        left.addWidget(btn_step3)

        btn_step4 = QPushButton("步骤4: 重构与分析")
        btn_step4.clicked.connect(self.step4_reconstruct)
        left.addWidget(btn_step4)

        btn_step5 = QPushButton("步骤5: FPR BPM")
        btn_step5.clicked.connect(self.step5_bpm)
        left.addWidget(btn_step5)

        btn_step6 = QPushButton("步骤6: PSD 分析")
        btn_step6.clicked.connect(self.step6_psd)
        left.addWidget(btn_step6)

        # 状态输出
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        left.addWidget(QLabel("输出:"))
        left.addWidget(self.text_edit)

        left_panel = QWidget()
        left_panel.setLayout(left)
        left_panel.setMaximumWidth(250)

        # 右侧 matplotlib 画布
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)

        layout.addWidget(left_panel)
        layout.addWidget(self.canvas)

    def log(self, msg):
        self.text_edit.append(msg)

    # --- 分步实现 ---
    def step1_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择数据文件", "../data/", "Text (*.txt)")
        if path:
            timestamps, frames = load_pressure_txt(path)
            timestamps, frames = clean_dataset(timestamps, frames)
            self.timestamps = timestamps
            self.frames = frames
            self.fs = 1.0 / np.mean(np.diff(timestamps))
            self.log(f"已加载: {frames.shape}, fs={self.fs:.2f} Hz")

    def step2_roi(self):
        self.signal_1d = get_dual_roi_mean(self.frames)
        self.log(f"ROI 提取完成, 信号长度: {len(self.signal_1d)}")
        # 绘制
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.signal_1d)
        ax.set_title("ROI 信号")
        self.canvas.draw()

    def step3_decompose(self):
        # 调用具体算法
        # self.components = your_decomposition(self.signal_1d, ...)
        self.log(f"分解完成")

    # ... step4, step5, step6 类似

    def closeEvent(self, event):
        # 恢复控制台输出 (如果有 TextRedirector)
        import sys
        sys.stdout = sys.__stdout__
        super().closeEvent(event)


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
```

### B. 控制台测试脚本

简单的批量测试，放在模块根目录（如 `HeartbeatRate/test_heartbeat.py`）：

```python
"""批量测试所有 {signal_type} 提取算法"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_pressure_txt
from preprocess import clean_dataset
from algorithms import extract_xxx1, extract_xxx2, ...


def test_all(data_path):
    timestamps, frames = load_pressure_txt(data_path)
    timestamps, frames = clean_dataset(timestamps, frames)
    fs = 1.0 / np.mean(np.diff(timestamps))

    algorithms = {
        "XXX1": extract_xxx1,
        "XXX2": extract_xxx2,
    }

    for name, func in algorithms.items():
        t0 = time.time()
        signal = func(frames, fs)
        elapsed = time.time() - t0
        bpm = calculate_bpm(signal, fs)
        print(f"{name:8s}: BPM={bpm:6.1f}, 耗时={elapsed:.2f}s")


if __name__ == "__main__":
    test_all("data/example.txt")
```

### C. 滑窗测试 GUI（复杂版本）

如果需要类似 `Main_test_UI.py` 或 `HeartbeatRate/main_UI.py` 的滑动窗口分析功能，关键要素：

```python
# 滑动窗口参数
self.window_sec = 30      # 窗口大小 (秒)
self.step_sec = 5         # 步长 (秒)
self.window_frames = int(self.window_sec * self.fs)
self.step_frames = int(self.step_sec * self.fs)

# 3 子图布局: BPM趋势 / 波形 / PSD
self.ax_bpm = self.figure.add_subplot(3, 1, 1)
self.ax_wave = self.figure.add_subplot(3, 1, 2)
self.ax_psd = self.figure.add_subplot(3, 1, 3)

# 键盘快捷键 (左右箭头翻窗)
self.setFocusPolicy(Qt.StrongFocus)

def keyPressEvent(self, event):
    if event.key() == Qt.Key_Right:
        self.current_window += 1
        self.update_plot()
    elif event.key() == Qt.Key_Left:
        self.current_window = max(0, self.current_window - 1)
        self.update_plot()

# TextRedirector 捕获 stdout 输出到 QTextEdit
class TextRedirector:
    def __init__(self, callback):
        self.callback = callback
    def write(self, text):
        self.callback(text)
    def flush(self):
        pass
```

---

## 任务三：创建全新的分析模块

当用户需要一个全新的分析模块（如新增"体动检测"、"呼吸暂停检测"、"姿势分类"等），按照以下模板创建：

### 目录结构

```
NewModuleName/
├── algorithms/
│   ├── __init__.py      # 算法注册表
│   ├── base.py          # 模块专用工具函数
│   └── xxx_extract.py   # 具体算法
├── main_UI.py           # 可选: GUI 界面
├── data_loader.py       # 复用 Breath_Extraction 的
├── preprocess.py        # 复用或定制
└── test_xxx.py          # 控制台测试
```

### `algorithms/__init__.py` 模板

```python
"""
NewModuleName 算法包

提供以下算法:
  - extract_xxx: 算法描述
"""

from .base import (
    butter_bandpass_filter,
    calculate_bpm,
    calculate_bpm_fpr,
    calculate_snr,
    # ... 其他共享函数
)

# 导入各算法入口
from .xxx1_extract import extract_xxx as extract_xxx1
from .xxx2_extract import extract_xxx as extract_xxx2

__all__ = [
    "extract_xxx1",
    "extract_xxx2",
    "butter_bandpass_filter",
    "calculate_bpm",
    "calculate_bpm_fpr",
    "calculate_snr",
]
```

### `algorithms/base.py` 所需工具函数

根据信号类型选择/调整参数：

| 参数 | 呼吸 | 心跳 | 通用 |
|------|------|------|------|
| 带通低截止 | 0.1 Hz | 0.8 Hz | - |
| 带通高截止 | 0.5 Hz | 2.2 Hz | - |
| 小波 alpha | 0.5 | 0.3 | 1.2 (强去噪) |
| 平滑窗口 | 41 | 7 | - |
| 寻峰最小间距 | fs*1.0 | fs*0.4 | - |
| SNR 频带 | [0.1, 0.4] | [0.8, 2.0] | - |

---

## 任务四：为算法添加命令行入口

如果需要独立运行的算法脚本（非 GUI），模式如下：

```python
"""
独立运行的 {Method} 呼吸/心跳提取脚本
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_pressure_txt
from preprocess import clean_dataset
from algorithms.base import get_dual_roi_mean, calculate_bpm, calculate_bpm_fpr


def main():
    # 加载数据
    data_path = sys.argv[1] if len(sys.argv) > 1 else "../data/example.txt"
    timestamps, frames = load_pressure_txt(data_path)
    timestamps, frames = clean_dataset(timestamps, frames)
    fs = 1.0 / np.mean(np.diff(timestamps))

    # 提取信号
    signal_1d = get_dual_roi_mean(frames)
    # ... 进一步处理 ...

    # 计算 BPM
    bpm = calculate_bpm(signal_1d, fs)
    bpm_fpr = calculate_bpm_fpr(signal_1d, fs)

    # 输出结果
    print(f"BPM (peak): {bpm:.1f}")
    print(f"BPM (FPR):  {bpm_fpr:.1f}")

    # 绘图保存
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    t = np.arange(len(signal_1d)) / fs
    axes[0].plot(t, signal_1d)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Extracted Signal")
    # PSD
    from scipy.signal import welch
    freqs, psd = welch(signal_1d, fs, nperseg=min(1024, len(signal_1d)))
    axes[1].semilogy(freqs, psd)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_title("Power Spectral Density")
    plt.tight_layout()
    plt.savefig("result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
```

---

## 注意事项与常见问题

### 已知 Bug
1. **Breath_Extraction/algorithms/`__init__.py` 第 24 行缺少逗号**，导致字符串拼接。新写 `__init__.py` 时务必检查逗号。

### 性能考虑
- SMVMD 很慢但效果好，适合离线分析
- VMD 中等速度，K 越大越慢
- EMD 最快但精度一般
- 实时场景用最简单的算法 + 较小的 K 值

### 依赖项
- PyQt5: GUI 框架
- numpy, scipy: 数值计算和信号处理
- matplotlib: 图表绘制
- pywt: 小波去噪 (PyWavelets)
- PyEMD: EMD 分解
- vmdpy: VMD 分解
- PySide6 (extract.py 使用，非主流)

### 数据格式
- TXT 文件每行 1025 个 uint16 值 (1024 压力值 + 1 时间戳)
- 压力值还原为 32×32 矩阵，逐行排列

---

## 交互流程

当用户请求生成新代码时，按以下流程操作：

1. **确认目标** — 问清楚用户想要什么：
   - 新算法？放在哪个模块下？
   - 新测试脚本？分步 GUI 还是控制台？
   - 全新模块？处理什么信号？

2. **选择模板** — 根据上述任务类型选择对应的模板

3. **适配参数** — 根据信号类型（呼吸/心跳/其他）调整滤波参数、函数签名等

4. **生成文件** — 创建文件并给出使用说明

5. **注册集成** — 如果是算法，提醒更新 `__init__.py` 和 `main_UI.py`
