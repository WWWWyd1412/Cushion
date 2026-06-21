"""
深色极简主题 QSS
=================
由 Breath_Extraction 和 HeartbeatRate 的 main_UI.py 中的重复 QSS 提取而来。
所有 PyQt5 GUI 入口统一使用此主题。
"""

DARK_THEME_QSS = """
    QMainWindow {
        background-color: #1a1a1e;
    }
    QGroupBox {
        background-color: #24242b;
        border: 1px solid #3f3f46;
        border-radius: 8px;
        margin-top: 10px;
        color: #e4e4e7;
        font-weight: bold;
        font-size: 13px;
        padding: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px;
        left: 10px;
    }
    QPushButton {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: bold;
        font-size: 13px;
    }
    QPushButton:hover {
        background-color: #60a5fa;
    }
    QPushButton:pressed {
        background-color: #2563eb;
    }
    QPushButton:disabled {
        background-color: #4b5563;
        color: #9ca3af;
    }
    QLabel {
        color: #e4e4e7;
        font-size: 13px;
    }
    QComboBox {
        background-color: #27272a;
        border: 1px solid #3f3f46;
        border-radius: 6px;
        padding: 5px;
        color: #f4f4f5;
        font-size: 13px;
    }
    QComboBox:disabled {
        background-color: #1f1f23;
        color: #71717a;
    }
    QLineEdit {
        background-color: #27272a;
        border: 1px solid #3f3f46;
        border-radius: 6px;
        padding: 5px;
        color: #f4f4f5;
        font-size: 13px;
    }
    QCheckBox {
        color: #e4e4e7;
        font-size: 13px;
    }
    QTextEdit {
        background-color: #121214;
        color: #10b981;
        border: 1px solid #3f3f46;
        border-radius: 8px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 12px;
    }
    QProgressBar {
        background-color: #27272a;
        border: 1px solid #3f3f46;
        border-radius: 6px;
        text-align: center;
        color: #e4e4e7;
        font-size: 12px;
    }
    QProgressBar::chunk {
        background-color: #3b82f6;
        border-radius: 5px;
    }
"""
