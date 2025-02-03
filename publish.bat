@echo off

:: Clean previous builds
rmdir /s /q dist build
del /s /q *.egg-info

:: Build the package
python -m build

:: Upload to PyPI
python -m twine upload dist/* 