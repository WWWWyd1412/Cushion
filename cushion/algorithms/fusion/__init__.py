"""
cushion.algorithms.fusion — 多通道融合算法
==========================================
- ICA: FastICA 盲源分离融合 (Multi-ROI ICA)
- PCA: 主成分分析融合 (降维保底方案)
"""

from .ica import fuse_signals_ica, extract_multi_roi_ica
from .pca import fuse_signals_pca

__all__ = [
    "fuse_signals_ica",
    "extract_multi_roi_ica",
    "fuse_signals_pca",
]
