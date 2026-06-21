# 柔性压力阵列坐垫监测系统 (Cushion v2.0)

基于 32×32 柔性压力传感器阵列的生理信号监测科研系统，实现**呼吸率**和**心率 (BCG)** 的非接触式提取。

---

## 项目架构

```
new_CUSHION/
├── cushion/                          # ★ 核心共享包 (v2.0)
│   ├── core/                         # 基础设施
│   │   ├── data_loader.py            # 统一 TXT 数据加载
│   │   ├── preprocessor.py           # 参数化帧预处理
│   │   └── signal_utils.py           # 滤波/去噪/SNR/BPM/平滑
│   ├── algorithms/                   # 信号分解与融合
│   │   ├── base.py                   # ROI 提取 + 分量选择 + 重构
│   │   ├── decomposition/            # EMD, VMD, SMVMD, MVMD, ACMD, VME
│   │   └── fusion/                   # FastICA, PCA
│   ├── breath/                       # 呼吸配置 (0.1-0.5 Hz)
│   ├── heartbeat/                    # 心跳配置 (0.8-2.2 Hz)
│   └── ui/                           # 共享 UI (主题/控件/滑窗引擎)
│
├── apps/                             # ★ 应用入口 (基于 cushion)
│   ├── breath_analyzer.py            # 呼吸滑窗分析 GUI (7 种算法)
│   └── heartbeat_analyzer.py         # 心跳滑窗分析 GUI (4 种算法)
│
├── scripts/                          # ★ CLI 批量测试
│   ├── analyze_breath.py             # 呼吸算法批量对比
│   └── analyze_heartbeat.py          # 心跳算法批量对比
│
├── Breath_Extraction/                # 呼吸模块 (旧版, 独立运行)
├── HeartbeatRate/                    # 心跳模块 (旧版, 独立运行)
├── Real_Time_Extraction/             # 实时采集模块
├── Test_4_4/                         # 4×4 串口调试工具
├── data/                             # TXT 压力矩阵数据
├── papers/                           # 文献综述
├── requirements.txt                  # 依赖声明
└── pyproject.toml                    # 包配置
```

---

## 数据流

```
32×32 压力传感器 → 串口 (460800bps) → heat.py 实时采集
                                           ↓
                                    存储为 .txt (时间戳 + 1024值/帧)
                                           ↓
                    ┌──────────────────────┴──────────────────────┐
                    ↓                                              ↓
          Breath_Extraction/main_UI.py                HeartbeatRate/main_UI.py
          或 apps/breath_analyzer.py                 或 apps/heartbeat_analyzer.py
                    ↓                                              ↓
         cushion.algorithms (EMD/VMD/...)           cushion.algorithms (EMD/VMD/...)
                    ↓                                              ↓
            呼吸波形 + BPM                                    心跳波形 + HR
```

---

## 算法全景

### 呼吸提取 (Breath Extraction) — 7 种算法

| 算法 | 描述 | 分量策略 | 速度 |
|------|------|---------|------|
| **EMD** | 经验模态分解 | 多分量 SNR 重构 | ⚡ 快 |
| **VMD** | 变分模态分解 (K=5, α=2000) | 多分量 SNR 重构 | ⚡ 中 |
| **AFD** | 自适应傅里叶分解 (Hilbert 模拟) | 多分量 SNR 重构 | ⚡ 快 |
| **VMD_FPR (MAPE)** | VMD + MAPE 自动 K 值寻优 | 多分量 SNR 重构 | 🐢 慢 |
| **GOA-VMD** | 蚱蜢优化 VMD 参数 (K, α) | 多分量 SNR 重构 | 🐢 慢 |
| **SMVMD** | 逐次多元 VMD (空间多通道) | SNR 门限融合 | ⚡ 中 |
| **MVMD** | 多元 VMD (时空联合分解) | FastICA + SNR | ⚡ 中 |
| **Multi-ROI ICA** | 4 象限 ROI + FastICA 盲源分离 | ICA 分量筛选 | ⚡ 快 |

### 心跳提取 (Heartbeat / BCG) — 4 种算法

| 算法 | 描述 | 分量策略 | 预处理 |
|------|------|---------|--------|
| **EMD** | 经验模态分解 | 单分量最优 | VME 基线去除 |
| **VMD** | 变分模态分解 (K=6, α=2000) | 单分量最优 | VME 基线去除 |
| **ACMD** | 自适应啁啾模式分解 | 单分量最优 | VME 基线去除 |
| **VME** | 变分模态提取 (单模态定位) | 直接提取主频 | VME 基线去除 |

---

## 关键参数

| 参数 | 呼吸 | 心跳 |
|------|------|------|
| **生理频带** | 0.1 – 0.5 Hz | 0.8 – 2.2 Hz |
| **采样率** | 10 Hz | 10 Hz |
| **VMD 模态数 K** | 5 | 6 |
| **小波去噪 α** | 0.5 | 0.3 |
| **Savitzky-Golay 窗口** | 41 | 7 |
| **BPM 峰值最小间距** | 1.5 s (≤40 BPM) | 0.4 s (≤150 BPM) |
| **滑窗默认** | 250 帧 (25s) / 步 50 帧 | 15s / 步 3s |
| **PSD 显示范围** | 0 – 1.5 Hz | 0 – 3.0 Hz |

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动呼吸分析 GUI
python apps/breath_analyzer.py

# 启动心跳分析 GUI
python apps/heartbeat_analyzer.py

# 命令行批量测试
python scripts/analyze_breath.py
python scripts/analyze_heartbeat.py
```

---

## 依赖项

```
numpy, scipy, matplotlib, PyQt5          # 核心
PyWavelets (pywt)                         # 小波去噪
PyEMD                                     # EMD
vmdpy                                     # VMD
scikit-learn                              # FastICA / PCA
pyserial, pyqtgraph                       # 实时采集 (可选)
```

---

## 更新日志

**v2.0** — 架构重构:
- 新增 `cushion/` 核心共享包，消除 3 个模块间的代码重复
- 参数化所有频带配置 (BreathConfig / HeartbeatConfig)
- 新增 `apps/` 应用入口和 `scripts/` CLI 测试脚本
- 统一 UI 主题 (`cushion/ui/theme.py`) 和滑窗引擎
- 新增 `requirements.txt` 和 `pyproject.toml`

**v1.5** — 新增 MVMD、Multi-ROI ICA 算法；优化 FastICA；首尾 20s 裁剪

**v1.4** — 新增 HeartbeatRate 心跳节律提取模块 (ACMD/VMD/EMD/VME)

**v1.3** — 新增 GOA-VMD 蚱蜢优化算法；升级为滑窗分析系统

**v1.2** — 新增 SMVMD、ACMD 测试方法

**v1.1** — 新增预处理 (小波去噪 + 分量选择)

**v1.0** — 初始版本：VMD 呼吸提取
