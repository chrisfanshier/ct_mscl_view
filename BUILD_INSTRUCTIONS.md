# Building the Executable

This folder contains scripts to build a standalone .exe file for the Core Summary Viewer application.

## Prerequisites

1. Install Python 3.8 or higher
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Building the .exe

### Option 1: Quick Build (Recommended)

Simply double-click `build_exe.bat` or run it from PowerShell:
```
.\build_exe.bat
```

This will create a single executable file in the `dist` folder.

### Option 2: Using the Spec File (Advanced)

If you need to customize the build, edit `CoreSummaryViewer.spec` and run:
```
pyinstaller CoreSummaryViewer.spec
```

## Output

After building, you'll find:
- **dist/CoreSummaryViewer.exe** - Your standalone executable
- **build/** - Temporary build files (can be deleted)

## Troubleshooting

- If the build fails, ensure all dependencies are installed: `pip install -r requirements.txt`
- For debugging, edit the spec file and change `console=False` to `console=True`
- If the .exe is too large, consider using `--onedir` instead of `--onefile` in build_exe.bat

## Distribution

The `CoreSummaryViewer.exe` file in the `dist` folder can be distributed to other Windows computers without requiring Python to be installed.
