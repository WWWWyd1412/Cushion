"""
cushion.algorithms.decomposition — 信号分解算法
==============================================
- EMD:  经验模态分解
- VMD:  变分模态分解
- SMVMD: 逐次多元 VMD
- MVMD:  多元 VMD
- ACMD:  自适应啁啾模式分解 (心跳专用)
- VME:   变分模态提取 (心跳专用)
"""

from .emd import extract_emd
from .vmd import extract_vmd
from .smvmd import extract_smvmd
from .mvmd import extract_mvmd
from .acmd import extract_acmd, ACMD_Core
from .vme import extract_vme, VME_Core

__all__ = [
    "extract_emd",
    "extract_vmd",
    "extract_smvmd",
    "extract_mvmd",
    "extract_acmd",
    "ACMD_Core",
    "extract_vme",
    "VME_Core",
]
