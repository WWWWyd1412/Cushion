# algorithms/__init__.py

from .emd_extract import extract_respiration as extract_emd
from .vmd_extract import extract_respiration as extract_vmd
from .afd_extract import extract_respiration as extract_afd
from .vmd_MAPE import extract_respiration as extract_vmd_fpr
# === 新增 SMVMD 算法接口挂载 ===
from .smvmd_extract import extract_respiration as extract_smvmd
# === 新增 MVMD 与 Multi-ROI ICA 算法接口挂载 ===
from .mvmd_extract import extract_respiration as extract_mvmd
from .multi_roi_ica_extract import extract_respiration as extract_multi_roi_ica

from .base import (
    smooth_respiration_signal,
    calculate_bpm,
    calculate_bpm_fpr
)

__all__ = [
    "extract_emd",
    "extract_vmd",
    "extract_afd",
    "extract_vmd_fpr",
    "extract_smvmd",  # 显式暴露
    "extract_mvmd",  # MVMD 自适应多通道提取
    "extract_multi_roi_ica",  # 纯多 ROI ICA 提取
    "smooth_respiration_signal",
    "calculate_bpm",
    "calculate_bpm_fpr",
    "reconstruct_top3_components_by_energy",
]