# Gesture-to-Text: Real-time Hand Gesture Recognition

A computer vision application that converts hand-drawn letters, numbers, and symbols into text in real-time using your camera. Draw characters in the air with your index finger and watch them appear as text!

## Features

- **Real-time Hand Tracking**: Uses MediaPipe to track your hand and index finger position
- **Air Drawing**: Draw characters in the air with your index finger
- **Multiple Recognition Methods**: 
  - OCR (Optical Character Recognition) using EasyOCR
  - Template matching for basic characters
  - Machine learning support (extensible)
- **Live Camera Feed**: See your hand movements and drawings overlaid on the camera feed
- **No Recording**: Processes live camera feed without saving any video data

## Requirements

- Python 3.8+
- Webcam
- Good lighting conditions for optimal hand tracking

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd /Users/george/Documents/Coding/gesture-to-text
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application**:
   ```bash
   python main.py
   ```

2. **How to use**:
   - Position your hand in front of the camera
   - **Extend your index finger** to start drawing
   - **Keep your index finger extended** while drawing characters
   - **Lower your finger** (make a fist or relax your hand) to stop drawing
   - **Wait 2 seconds** after completing a character for automatic recognition
   - The recognized text will appear in the terminal and on screen

3. **Controls**:
   - `q`: Quit the application
   - `c`: Clear the recognized text

## Supported Characters

The application can recognize:
- **Letters**: A-Z (uppercase)
- **Numbers**: 0-9
- **Basic symbols**: `.`, `,`, `!`, `?`, `-`, `+`

## Tips for Best Results

1. **Good Lighting**: Ensure your environment is well-lit
2. **Clear Background**: Use a plain background behind your hand
3. **Steady Movements**: Draw characters slowly and clearly
4. **Finger Position**: Keep your index finger clearly extended while drawing
5. **Size**: Draw characters large enough to be clearly visible
6. **Completion**: Wait for the 2-second timeout before drawing the next character

## Project Structure

```
gesture-to-text/
├── main.py                    # Main application entry point
├── hand_tracker.py           # Hand tracking and gesture detection
├── drawing_manager.py        # Stroke tracking and drawing management
├── character_recognizer.py   # Character recognition from drawn strokes
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Architecture

### Hand Tracking (`hand_tracker.py`)
- Uses MediaPipe for real-time hand detection and landmark tracking
- Detects finger positions and gestures
- Determines when the index finger is extended for drawing

### Drawing Management (`drawing_manager.py`)
- Tracks finger movements to create stroke paths
- Smooths drawing lines using moving averages
- Manages stroke completion and rendering
- Converts drawings to images for recognition

### Character Recognition (`character_recognizer.py`)
- **OCR Method**: Uses EasyOCR for character recognition
- **Template Matching**: Basic pattern matching for simple characters
- **ML Ready**: Extensible framework for custom machine learning models
- Preprocesses drawn images for optimal recognition

### Main Application (`main.py`)
- Coordinates all components
- Manages the camera feed and real-time processing
- Handles user interface and controls
- Displays results and recognized text

## Troubleshooting

### Camera Issues
- Ensure your camera is not being used by another application
- Check camera permissions in your system settings
- Try changing the camera index in `main.py` if you have multiple cameras

### Hand Tracking Issues
- Ensure good lighting conditions
- Keep your hand clearly visible to the camera
- Avoid complex backgrounds that might interfere with hand detection

### Recognition Issues
- Draw characters clearly and at a reasonable size
- Ensure you wait for the 2-second timeout between characters
- Try drawing characters more slowly and deliberately

### Performance Issues
- Close other applications that might be using your camera or CPU
- Reduce the camera resolution if needed (modify `main.py`)

## Extending the Application

### Adding New Characters
1. Modify `character_recognizer.py` to add new templates
2. Update the `_is_valid_character()` method to include new characters
3. Train custom ML models with your own datasets

### Improving Recognition
1. Collect training data for machine learning models
2. Implement custom preprocessing techniques
3. Add more sophisticated feature extraction methods

### Adding New Gestures
1. Modify `hand_tracker.py` to detect new hand gestures
2. Add gesture-based commands (e.g., gestures for space, backspace, etc.)

## Dependencies

- `opencv-python`: Computer vision and image processing
- `mediapipe`: Hand tracking and pose estimation
- `numpy`: Numerical computations
- `Pillow`: Image processing utilities
- `tensorflow`: Machine learning framework
- `scikit-learn`: Machine learning algorithms
- `matplotlib`: Plotting and visualization
- `easyocr`: Optical character recognition
- `imutils`: Computer vision utilities

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Future Enhancements

- [ ] Support for lowercase letters
- [ ] Word and sentence recognition
- [ ] Custom gesture commands
- [ ] Multiple language support
- [ ] Mobile app version
- [ ] Voice feedback
- [ ] Gesture-based text editing (backspace, space, etc.)
- [ ] Export recognized text to file
