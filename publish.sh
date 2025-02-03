#!/bin/bash

# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build the package
python -m build

# Upload to PyPI
python -m twine upload dist/* 