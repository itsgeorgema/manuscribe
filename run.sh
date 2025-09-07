#!/bin/bash
# This script activates the virtual environment and runs the application

cd "$(dirname "$0")"
source venv/bin/activate

echo "Starting Gesture-to-Text Application..."
echo "Make sure your camera is connected and not in use by other applications."
echo "Press Ctrl+C to stop the application."
echo ""

python main.py
