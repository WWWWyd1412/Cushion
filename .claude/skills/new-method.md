---
name: new-method
description: 为新 CUSHION 项目生成新算法方法、测试脚本或新分析模块。当用户说"添加新算法"、"创建新方法"、"写个测试脚本"、"新建一个分析模块"时触发。覆盖在 cushion/algorithms/ 或旧模块中添加代码，以及在 scripts/ 创建 CLI 测试。
---

# 新 CUSHION 项目代码生成技能

这个技能帮助你按照项目既定规范和模式，在 new_CUSHION 项目中生成新的算法方法、测试脚本或全新的分析模块。

## 项目架构总览 (v2.0)

```
new_CUSHION/
├── cushion/                          # ★ 核心共享包 (v2.0, 消除代码重复)
│   ├── core/                         # 基础设施 (参数化, 3→1)
│   │   ├── data_loader.py            # 统一 TXT 数据加载
│   │   ├── preprocessor.py           # 参数化帧预处理
│   │   └── signal_utils.py           # 滤波/去噪/SNR/BPM/平滑 (参数化频带)
│   ├── algorithms/                   # 信号分解与融合
│   │   ├── base.py                   # ROI 提取 + 分量选择 + 重构 (参数化频带)
│   │   ├── decomposition/            # EMD, VMD, SMVMD, MVMD, ACMD, VME
│   │   └── fusion/                   # FastICA (Multi-ROI), PCA
│   ├── breath/config.py              # 呼吸配置: FREQ_BAND=(0.1,0.5), SAVGOL=(41,3)
│   ├── heartbeat/config.py           # 心跳配置: FREQ_BAND=(0.8,2.2), SAVGOL=(7,2)
│   └── ui/                           # 共享 PyQt5 UI 组件
│       ├── theme.py                  # 深色主题 QSS
│       ├── widgets.py                # TextRedirector
│       └── sliding_window.py         # 滑窗 + 重叠相加 + 相位对齐
│
├── apps/                             # ★ 应用入口 (基于 cushion 包)
│   ├── breath_analyzer.py            # 呼吸滑窗分析 GUI (7 种算法)
│   └── heartbeat_analyzer.py         # 心跳滑窗分析 GUI (4 种算法)
│
├── scripts/                          # ★ CLI 批量测试
│   ├── analyze_breath.py             # 呼吸算法批量对比
│   └── analyze_heartbeat.py          # 心跳算法批量对比
│
├── Breath_Extraction/                # 呼吸模块 (旧版, 独立运行, 保留兼容)
├── HeartbeatRate/                    # 心跳模块 (旧版, 独立运行, 保留兼容)
├── Real_Time_Extraction/             # 实时采集模块
├── Test_4_4/                         # 4×4 串口调试
├── data/                             # TXT 压力矩阵数据
├── papers/                           # 文献综述
├── requirements.txt                  # 依赖声明
└── pyproject.toml                    # 包配置
```

## 两种代码路径

### 路径 A: 在 cushion 包中添加 (推荐)

适用于新算法、新工具函数。代码放在 `cushion/` 下，所有模块共享。

**优点**: 一次编写，呼吸和心跳都能用（通过参数化频带）。

### 路径 B: 在旧模块中添加

适用于仅特定模块使用的代码。放在 `Breath_Extraction/`、`HeartbeatRate/` 等原有目录。

**使用场景**: 仅影响一个模块、不需要参数化。

---

## 核心约定

### 命名规范
- 分解算法: `cushion/algorithms/decomposition/{method}.py`
- 融合算法: `cushion/algorithms/fusion/{method}.py`
- 入口函数: `extract_{method}(frames, fs, **kwargs) -> np.ndarray`
- CLI 脚本: `scripts/analyze_{domain}.py`
- 辅助函数使用 `snake_case`

### 数据流
```
TXT 文件
  → cushion.core.data_loader.load_pressure_txt()
  → cushion.core.preprocessor.clean_dataset()
  → cushion.algorithms.base.get_dual_roi_mean()   # 或 get_spatial_sum()
  → cushion.algorithms.decomposition.{method}.extract_{method}()
  → cushion.core.signal_utils.calculate_bpm_peak() / calculate_bpm_fpr()
  → BPM
```

### 导入模式

```python
# === 在 cushion 包内部 (推荐) ===
from cushion.core import load_pressure_txt, clean_dataset
from cushion.core.signal_utils import (
    butter_bandpass_filter, wavelet_denoise, calculate_snr,
    smooth_signal, calculate_bpm_peak, calculate_bpm_fpr
)
from cushion.algorithms.base import (
    get_dual_roi_mean, get_multi_roi_signals, get_spatial_sum,
    select_best_component, reconstruct_multicomponent_with_snr,
    reconstruct_top3_by_energy
)
from cushion.algorithms.fusion.ica import fuse_signals_ica
from cushion.breath.config import BreathConfig    # 呼吸参数
from cushion.heartbeat.config import HeartbeatConfig  # 心跳参数
from cushion.ui.theme import DARK_THEME_QSS
from cushion.ui.widgets import TextRedirector
from cushion.ui.sliding_window import generate_windows, compute_bpm_statistics

# === Conda DLL 路径修复 (所有 GUI 文件开头) ===
_conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)
_qt_dll_path = os.path.join(_conda_prefix, 'Library', 'bin')
if os.path.isdir(_qt_dll_path) and _qt_dll_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _qt_dll_path + os.pathsep + os.environ.get('PATH', '')

# === matplotlib 中文字体配置 ===
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
```

---

## 任务一：在 cushion 中添加新分解算法

### 第一步：确定信号域和参数

| 域 | 频带 | VMD K | 小波 α | 分量策略 | 基线去除 |
|----|------|-------|--------|---------|---------|
| 呼吸 | [0.1, 0.5] Hz | 5 | 0.5 | 多分量 SNR | 否 |
| 心跳 | [0.8, 2.2] Hz | 6 | 0.3 | 单分量最优 | 是 (VME) |

### 第二步：创建 `cushion/algorithms/decomposition/{method}.py`

```python
"""
{Method} 信号提取
=================
"""

import numpy as np
from cushion.algorithms.base import (
    get_dual_roi_mean, get_spatial_sum,
    select_best_component, reconstruct_multicomponent_with_snr,
)


def extract_{method}(frames, fs,
                     freq_band=(0.1, 0.5),
                     wavelet_alpha=0.5,
                     use_vme_baseline=False,
                     use_multicomponent=True,
                     roi_mode='dual'):
    """
    {Method} 信号提取。

    Parameters
    ----------
    frames : ndarray (N, 32, 32)
    fs : float
    freq_band : tuple — 目标频带 (low, high) Hz
    wavelet_alpha : float — 小波去噪强度
    use_vme_baseline : bool — 心跳模式: 启用 VME 基线漂移去除
    use_multicomponent : bool — True: 多分量SNR, False: 单分量最优
    roi_mode : str — 'dual' (Breath/Heartbeat) 或 'spatial_sum' (RealTime)

    Returns
    -------
    ndarray — 1D 提取信号
    """
    # 1. ROI 提取
    if roi_mode == 'spatial_sum':
        signal_1d = get_spatial_sum(frames)
    else:
        signal_1d = get_dual_roi_mean(
            frames, fs=fs, freq_band=freq_band,
            wavelet_alpha=wavelet_alpha,
            use_vme_baseline=use_vme_baseline)

    if len(signal_1d) == 0:
        return np.zeros(100)

    # 2. 核心分解逻辑
    components = your_decomposition(signal_1d, ...)

    # 3. 分量选择与重构
    if use_multicomponent:
        return reconstruct_multicomponent_with_snr(components, fs, freq_band=freq_band)
    return select_best_component(components, fs, freq_band=freq_band)
```

### 第三步：注册算法

在 `cushion/algorithms/decomposition/__init__.py`:
```python
from .{method} import extract_{method}

__all__ = [..., "extract_{method}"]
```

在 `cushion/algorithms/__init__.py`:
```python
from .decomposition.{method} import extract_{method}
```

### 第四步：在 apps 中注册

在 `apps/breath_analyzer.py` 的 `ALGO_REGISTRY` 添加:
```python
"{METHOD_DISPLAY}": lambda frames, fs: extract_{method}(
    frames, fs, freq_band=CFG.FREQ_BAND, wavelet_alpha=CFG.WAVELET_ALPHA),
```

同时在 `algo_selector.addItems([...])` 中添加算法名。

---

## 任务二：创建测试脚本

### A. CLI 批量测试 (推荐)

放在 `scripts/` 目录:

```python
#!/usr/bin/env python3
"""批量测试 {signal_type} 提取算法"""
import sys, os, time
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from cushion.core import load_pressure_txt, clean_dataset
from cushion.core.signal_utils import smooth_signal, calculate_bpm_peak, calculate_bpm_fpr
from cushion.algorithms.decomposition.emd import extract_emd
from cushion.algorithms.decomposition.vmd import extract_vmd
from cushion.breath.config import BreathConfig as CFG
from cushion.ui.sliding_window import generate_windows, compute_bpm_statistics


DATA_PATH = os.path.join(_project_root, "data", "20260501_162541.txt")

def main():
    # 加载 + 清洗
    timestamps, frames = load_pressure_txt(DATA_PATH)
    clean_times, clean_frames = clean_dataset(timestamps, frames, ...)

    # 滑窗分析
    windows = generate_windows(len(clean_frames), window_size, step_size)

    for algo_name, extract_fn in algorithms.items():
        bpm_list = []
        for start, end in windows:
            wave = extract_fn(clean_frames[start:end])
            wave = smooth_signal(wave, ...)
            bpm = calculate_bpm_peak(wave, FS)
            bpm_list.append(bpm)

        stats = compute_bpm_statistics(bpm_list)
        print(f"{algo_name}: mean={stats['mean']:.1f}, ...")

if __name__ == "__main__":
    sys.exit(main())
```

### B. 分步验证 GUI (旧模块)

放在 `Breath_Extraction/algorithms/test_scripts/`，参考现有的 `vmd_fpr_standalone.py`、`smvmd_fpr_standalone.py`。

---

## 任务三：创建全新的分析模块

在 cushion 包中添加新域（如姿势分类、体动检测）:

```
cushion/{new_domain}/
├── __init__.py
└── config.py          # 域专用参数
```

然后在 `apps/` 创建对应的 GUI 入口。

---

## 关键参数速查

| 参数 | BreathConfig | HeartbeatConfig |
|------|-------------|-----------------|
| FREQ_BAND | (0.1, 0.5) | (0.8, 2.2) |
| SNR_BAND | (0.1, 0.4) | (0.8, 2.2) |
| WAVELET_ALPHA | 0.5 | 0.3 |
| SAVGOL_WINDOW | 41 | 7 |
| SAVGOL_ORDER | 3 | 2 |
| VMD_K | 5 | 6 |
| VMD_ALPHA | 2000 | 2000 |
| BPM_MIN_DIST_SEC | 1.5 | 0.4 |
| BPM_PROMINENCE_RATIO | 0.5 | 0.15 |
| GAUSSIAN_SIGMA | 0.8 | 0.0 (不启用) |
| USE_VME_BASELINE | False | True |
| PSD xlim | (0, 1.5) | (0, 3.0) |

## 依赖项

```
numpy, scipy, matplotlib, PyQt5          # 核心
PyWavelets (pywt)                         # 小波去噪
PyEMD                                     # EMD
vmdpy                                     # VMD
scikit-learn                              # FastICA / PCA
pyserial, pyqtgraph                       # 实时采集 (可选)
```

## 注意事项

1. **参数化优先**: 新算法应接受 `freq_band`、`wavelet_alpha` 等参数，而非硬编码
2. **配置类**: 使用 `BreathConfig` / `HeartbeatConfig` 获取默认参数
3. **信号工具**: 所有滤波/去噪/BPM 函数在 `cushion.core.signal_utils` 中统一管理
4. **中文字体**: 所有 GUI 文件必须配置 `matplotlib.rcParams['font.sans-serif']`
5. **Conda DLL**: GUI 文件必须包含 Qt DLL 路径修复
6. **旧模块兼容**: `Breath_Extraction/`、`HeartbeatRate/` 仍可独立运行，但新代码优先放在 `cushion/`

## 交互流程

1. **确认目标** — 新算法？新测试？新模块？放在 cushion 还是旧模块？
2. **选择模板** — 根据任务类型选择对应模板
3. **适配参数** — 根据信号域（呼吸/心跳）选择配置
4. **生成文件** — 创建文件
5. **注册集成** — 更新 `__init__.py` 和 `apps/` 中的算法注册表