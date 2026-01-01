@echo off
echo Testing hello_world.py...
python hello_world.py
if %errorlevel% equ 0 (
    echo Test passed!
) else (
    echo Test failed with error code %errorlevel%
    exit /b 1
)