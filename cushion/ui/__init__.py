"""
cushion.ui — 共享 UI 组件
=========================
提供跨模块复用的 PyQt5 界面组件。

- theme: 深色极简主题 QSS
- widgets: 可复用控件 (日志重定向器)
- sliding_window: 滑窗分析引擎 (重叠相加融合 + 相位对齐)
"""

from .theme import DARK_THEME_QSS
from .widgets import TextRedirector

__all__ = [
    "DARK_THEME_QSS",
    "TextRedirector",
]
