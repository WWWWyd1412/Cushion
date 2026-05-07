# algorithms/__init__.py

# 从子模块中导入核心提取函数
from .vmd_MAPE import extract_respiration as extract_vmd_fpr
from .emd_extract import extract_respiration as extract_emd
from .vmd_extract import extract_respiration as extract_vmd
from .afd_extract import extract_respiration as extract_afd
from .base import calculate_bpm, calculate_bpm_fpr, smooth_respiration_signal


# 定义包被 * 导入时可见的内容
__all__ = ['extract_emd', 'extract_vmd', 'extract_vmd_fpr', 'extract_afd', 'calculate_bpm', 'calculate_bpm_fpr', 'smooth_respiration_signal', 'optimize_vmd_with_mape']