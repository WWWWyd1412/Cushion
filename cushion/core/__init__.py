"""
cushion.core — 共享基础设施
============================
提供跨模块复用的数据加载、预处理和信号处理工具。
"""

from .data_loader import load_pressure_txt, get_session_info
from .preprocessor import Preprocessor, clean_dataset
from .signal_utils import (
    butter_bandpass_filter,
    wavelet_denoise,
    calculate_snr,
    smooth_signal,
)

__all__ = [
    "load_pressure_txt",
    "get_session_info",
    "Preprocessor",
    "clean_dataset",
    "butter_bandpass_filter",
    "wavelet_denoise",
    "calculate_snr",
    "smooth_signal",
]
