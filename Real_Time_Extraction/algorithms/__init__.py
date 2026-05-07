# algorithms/__init__.py
from .emd_extract import extract_respiration as extract_emd
from .vmd_extract import extract_respiration as extract_vmd
from .afd_extract import extract_respiration as extract_afd
from .base import calculate_bpm, get_spatial_sum, smooth_signal

__all__ = [
    'extract_emd',
    'extract_vmd',
    'extract_afd',
    'calculate_bpm',
    'get_spatial_sum',
    'smooth_signal'
]