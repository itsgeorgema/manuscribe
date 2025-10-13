#!/usr/bin/env python3
"""
Manuscribe: Real-time hand gesture recognition for text input
"""

import cv2
import numpy as np
import time

from hand_tracker import HandTracker
from drawing_manager import DrawingManager
from character_recognizer import CharacterRecognizer


class GestureToTextApp:
    def __init__(self):
        """Initialize the Manuscribe application."""
        # Use system default camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize components
        self.hand_tracker = HandTracker()
        self.drawing_manager = DrawingManager()
        self.character_recognizer = CharacterRecognizer()
        
        # Application state
        self.is_running = True
        self.recognized_text = ""
        self.current_drawing = []
        self.last_finger_pos = None
        self.drawing_timeout = 2.0  # seconds to wait before processing drawing
        self.last_drawing_time = time.time()
        
        # UI elements
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_thickness = 2
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame from the camera."""
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands and get finger positions
        hands = self.hand_tracker.detect_hands(frame_rgb)
        
        if hands:
            # Get index finger tip position
            finger_pos = self.hand_tracker.get_index_finger_tip(hands[0])
            
            if finger_pos:
                # Convert normalized coordinates to pixel coordinates
                h, w, _ = frame.shape
                finger_pixel_pos = (int(finger_pos[0] * w), int(finger_pos[1] * h))
                
                # Check if finger is in drawing position (extended)
                if self.hand_tracker.is_finger_extended(hands[0]):
                    # Add point to current drawing
                    if self.last_finger_pos is not None:
                        self.drawing_manager.add_stroke_point(finger_pixel_pos)
                    self.last_finger_pos = finger_pixel_pos
                    self.last_drawing_time = time.time()
                else:
                    # Finger not extended, stop drawing
                    if self.last_finger_pos is not None:
                        self.drawing_manager.end_current_stroke()
                    self.last_finger_pos = None
                
                # Draw finger position indicator
                cv2.circle(frame, finger_pixel_pos, 8, (0, 255, 0), -1)
        else:
            self.last_finger_pos = None
        
        # Draw current strokes on frame
        frame = self.drawing_manager.draw_strokes(frame)
        
        # Check if we should process the drawing (timeout reached)
        if (time.time() - self.last_drawing_time > self.drawing_timeout and 
            self.drawing_manager.has_strokes()):
            self.process_drawing()
        
        # Add UI elements
        frame = self.draw_ui(frame)
        
        return frame
    
    def process_drawing(self):
        """Process the current drawing to recognize characters."""
        drawing_image = self.drawing_manager.get_drawing_as_image()
        
        if drawing_image is not None:
            print(f"Processing drawing image with shape: {drawing_image.shape}")
            print(f"Drawing brightness: {drawing_image.mean():.2f}")
            
            # Recognize character from drawing
            recognized_char = self.character_recognizer.recognize(drawing_image)
            
            if recognized_char and recognized_char.strip():
                self.recognized_text += recognized_char
                print(f"Recognized: {recognized_char}")
                print(f"Current text: {self.recognized_text}")
            else:
                print("No character recognized")
        else:
            print("No drawing to process")
        
        # Clear the drawing
        self.drawing_manager.clear_drawing()
    
    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI elements on the frame."""
        h, w, _ = frame.shape
        
        # Draw background rectangle for text
        text_bg_height = 100
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, text_bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw recognized text
        text = f"Recognized Text: {self.recognized_text}"
        cv2.putText(frame, text, (10, 30), self.font, 0.8, (255, 255, 255), 2)
        
        # Draw instructions
        instructions = [
            "Instructions:",
            "- Point with index finger to draw",
            "- Keep finger extended while drawing",
            "- Lower finger to stop drawing",
            "- Wait 2 seconds after drawing for recognition",
            "- Press 'q' to quit, 'c' to clear text"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = h - (len(instructions) - i) * 25 - 10
            cv2.putText(frame, instruction, (10, y_pos), self.font, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main application loop."""
        print("Manuscribe Started")
        print("Point with your index finger to draw letters, numbers, or symbols")
        print("Press 'q' to quit, 'c' to clear recognized text")
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Display the frame
            cv2.imshow('Gesture to Text', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.is_running = False
            elif key == ord('c'):
                self.recognized_text = ""
                self.drawing_manager.clear_drawing()
                print("Cleared recognized text")
            # Camera cycling disabled; always use system default
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"Final recognized text: {self.recognized_text}")


if __name__ == "__main__":
    app = GestureToTextApp()
    app.run()
