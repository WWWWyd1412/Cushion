"""
共享可复用控件
==============
- TextRedirector: stdout 重定向到 QTextEdit 日志面板
"""


class TextRedirector:
    """
    捕获 stdout print 输出，显示在 UI 日志面板中。

    用法:
        sys.stdout = TextRedirector(self.log_panel.append)
    """

    def __init__(self, write_func):
        """
        Parameters
        ----------
        write_func : callable
            接受一个字符串参数的回调函数，用于将日志写入 UI 控件。
        """
        self.write_func = write_func

    def write(self, text):
        if text.strip():
            self.write_func(text.strip())

    def flush(self):
        pass
