"""
Cushion — 柔性压力阵列坐垫生理信号监测框架
==============================================
提供数据加载、预处理、信号分解算法、UI组件的统一接口。

模块结构:
    cushion.core       — 共享基础设施 (data_loader, preprocessor, signal_utils)
    cushion.algorithms — 信号分解与融合算法 (EMD, VMD, SMVMD, MVMD, ACMD, VME, ICA, PCA)
    cushion.breath     — 呼吸分析专用配置
    cushion.heartbeat  — 心跳分析专用配置
    cushion.ui         — 共享 UI 组件 (主题, 控件, 滑窗逻辑)
"""

__version__ = "2.0.0"
