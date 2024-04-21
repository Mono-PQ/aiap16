#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment 
source venv/bin/activate

# Install requirements in requirements.txt
pip install -r requirements.txt

# Stop script if an error occurs
set -e

# Run main.py
python main.py

# Deactivate virtual environment
deactivate
