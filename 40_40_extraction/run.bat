@echo off
call D:\anaconda\Scripts\activate.bat torch
python "%~dp0main.py" %*
