# algorithms/__init__.py

from .emd_extract import extract_respiration as extract_emd
from .vmd_extract import extract_respiration as extract_vmd
from .afd_extract import extract_respiration as extract_afd
from .vmd_MAPE import extract_respiration as extract_vmd_fpr
# === 新增 SMVMD 算法接口挂载 ===
from .smvmd_extract import extract_respiration as extract_smvmd

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
    "smooth_respiration_signal",
    "calculate_bpm",
    "calculate_bpm_fpr"
]