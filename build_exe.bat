@echo off
REM PyInstaller build script for Core Summary Viewer
REM This will create a standalone .exe file

echo Building Core Summary Viewer executable...
echo.

pyinstaller --noconfirm ^
  --name="CoreSummaryViewer" ^
  --windowed ^
  --onefile ^
  --icon=NONE ^
  --add-data="tiff_collect.py;." ^
  --hidden-import=PySide6.QtCore ^
  --hidden-import=PySide6.QtGui ^
  --hidden-import=PySide6.QtWidgets ^
  --hidden-import=PIL._tkinter_finder ^
  --collect-all=reportlab ^
  tiff_gui.py

echo.
echo Build complete! Your .exe file is in the 'dist' folder.
echo.
pause
