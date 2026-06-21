"""
cushion.algorithms — 信号分解与融合算法
========================================
统一接口: 所有提取函数签名为 extract(frames, fs, **kwargs) -> 1D signal。

分解算法:
    - EMD  (经验模态分解)
    - VMD  (变分模态分解)
    - SMVMD (逐次多元VMD)
    - MVMD  (多元VMD)
    - ACMD  (自适应啁啾模式分解)
    - VME   (变分模态提取)

融合算法:
    - Multi-ROI ICA (多区域独立成分分析)
    - PCA            (主成分分析融合)
"""

from .decomposition.emd import extract_emd
from .decomposition.vmd import extract_vmd
from .decomposition.smvmd import extract_smvmd
from .decomposition.mvmd import extract_mvmd
from .decomposition.acmd import extract_acmd
from .decomposition.vme import extract_vme
from .fusion.ica import extract_multi_roi_ica
from .fusion.pca import fuse_signals_pca

__all__ = [
    "extract_emd",
    "extract_vmd",
    "extract_smvmd",
    "extract_mvmd",
    "extract_acmd",
    "extract_vme",
    "extract_multi_roi_ica",
    "fuse_signals_pca",
]
