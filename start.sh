#!/bin/bash

# Add the directory to the PATH
export PATH="/opt/render/.local/bin:$PATH"

# Upgrade pip and install packages from requirements.txt
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Start your Flask application
python app.py
