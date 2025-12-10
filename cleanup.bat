@echo off
REM Cleanup script for tide level image processing project
REM Removes debug, output, processed, and log directories for clean testing

echo Starting cleanup of tide level processing directories...

set directories=data\debug data\output data\processed logs

for %%d in (%directories%) do (
    if exist "%%d" (
        echo Clearing contents of directory: %%d
        del /s /q "%%d\*" >nul 2>&1
        for /d %%x in ("%%d\*") do rmdir /s /q "%%x" >nul 2>&1
        echo Cleared contents of %%d
    ) else (
        echo Error: Directory not found: %%d
    )
)

echo.
echo Cleanup completed! Ready for fresh testing.
echo Note: Input and calibration directories were preserved.