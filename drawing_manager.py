"""
Drawing management module for tracking and rendering finger strokes.
"""

import cv2
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict
import time


class Stroke:
    """Represents a single drawing stroke."""
    
    def __init__(self, color: Tuple[int, int, int] = (0, 255, 255), thickness: int = 3):
        """
        Initialize a stroke.
        
        Args:
            color: RGB color tuple for the stroke
            thickness: Thickness of the stroke line
        """
        self.points: List[Tuple[int, int]] = []
        self.color = color
        self.thickness = thickness
        self.timestamp = time.time()
        self.is_complete = False
    
    def add_point(self, point: Tuple[int, int]):
        """Add a point to the stroke."""
        self.points.append(point)
    
    def complete(self):
        """Mark the stroke as complete."""
        self.is_complete = True
    
    def get_bounding_box(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the bounding box of the stroke.
        
        Returns:
            (x, y, width, height) or None if no points
        """
        if not self.points:
            return None
        
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)


class DrawingManager:
    """Manages drawing strokes and rendering."""
    
    def __init__(self, 
                 canvas_size: Tuple[int, int] = (1280, 720),
                 stroke_color: Tuple[int, int, int] = (0, 255, 255),
                 stroke_thickness: int = 3,
                 min_stroke_length: int = 5):
        """
        Initialize the drawing manager.
        
        Args:
            canvas_size: Size of the drawing canvas (width, height)
            stroke_color: Default color for strokes (B, G, R for OpenCV)
            stroke_thickness: Default thickness for strokes
            min_stroke_length: Minimum number of points for a valid stroke
        """
        self.canvas_size = canvas_size
        self.stroke_color = stroke_color
        self.stroke_thickness = stroke_thickness
        self.min_stroke_length = min_stroke_length
        
        # Drawing state
        self.strokes: List[Stroke] = []
        self.current_stroke: Optional[Stroke] = None
        self.drawing_canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
        
        # Smoothing parameters
        self.smoothing_window = 3
        self.point_history = deque(maxlen=self.smoothing_window)
        
    def add_stroke_point(self, point: Tuple[int, int]):
        """
        Add a point to the current stroke.
        
        Args:
            point: (x, y) coordinates of the point
        """
        # Smooth the point using moving average
        self.point_history.append(point)
        
        if len(self.point_history) >= 2:
            smoothed_point = self._smooth_point()
            
            # Create new stroke if needed
            if self.current_stroke is None:
                self.current_stroke = Stroke(self.stroke_color, self.stroke_thickness)
            
            # Add smoothed point to current stroke
            self.current_stroke.add_point(smoothed_point)
    
    def _smooth_point(self) -> Tuple[int, int]:
        """Apply smoothing to the current point based on recent history."""
        if len(self.point_history) == 0:
            return (0, 0)
        
        # Calculate weighted average of recent points
        weights = np.linspace(0.5, 1.0, len(self.point_history))
        weights = weights / np.sum(weights)
        
        avg_x = sum(p[0] * w for p, w in zip(self.point_history, weights))
        avg_y = sum(p[1] * w for p, w in zip(self.point_history, weights))
        
        return (int(avg_x), int(avg_y))
    
    def end_current_stroke(self):
        """End the current stroke and add it to the strokes list."""
        if self.current_stroke is not None and len(self.current_stroke.points) >= self.min_stroke_length:
            self.current_stroke.complete()
            self.strokes.append(self.current_stroke)
        
        self.current_stroke = None
        self.point_history.clear()
    
    def draw_strokes(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw all strokes on the given frame.
        
        Args:
            frame: Input frame to draw on
            
        Returns:
            Frame with strokes drawn
        """
        frame_copy = frame.copy()
        
        # Draw completed strokes
        for stroke in self.strokes:
            self._draw_stroke(frame_copy, stroke)
        
        # Draw current stroke
        if self.current_stroke is not None:
            self._draw_stroke(frame_copy, self.current_stroke)
        
        return frame_copy
    
    def _draw_stroke(self, frame: np.ndarray, stroke: Stroke):
        """Draw a single stroke on the frame."""
        if len(stroke.points) < 2:
            return
        
        # Draw lines between consecutive points
        for i in range(1, len(stroke.points)):
            cv2.line(frame, stroke.points[i-1], stroke.points[i], 
                    stroke.color, stroke.thickness)
        
        # Draw circles at stroke endpoints for better visibility
        if stroke.points:
            cv2.circle(frame, stroke.points[0], 2, (0, 255, 0), -1)  # Green start
            if stroke.is_complete:
                cv2.circle(frame, stroke.points[-1], 2, (0, 0, 255), -1)  # Red end
    
    def get_drawing_as_image(self, 
                            padding: int = 20, 
                            target_size: Tuple[int, int] = (128, 128)) -> Optional[np.ndarray]:
        """
        Get the current drawing as a processed image for character recognition.
        
        Args:
            padding: Padding around the drawing bounding box
            target_size: Target size for the output image
            
        Returns:
            Processed image or None if no drawing exists
        """
        if not self.strokes:
            return None
        
        # Create a clean canvas
        canvas = np.zeros(self.canvas_size[::-1], dtype=np.uint8)  # (height, width)
        
        # Draw all strokes on the canvas
        for stroke in self.strokes:
            if len(stroke.points) >= 2:
                for i in range(1, len(stroke.points)):
                    cv2.line(canvas, stroke.points[i-1], stroke.points[i], 255, 3)
        
        # Find bounding box of all strokes
        bbox = self._get_all_strokes_bounding_box()
        if bbox is None:
            return None
        
        x, y, w, h = bbox
        
        # Add padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(self.canvas_size[0] - x, w + 2 * padding)
        h = min(self.canvas_size[1] - y, h + 2 * padding)
        
        # Crop the drawing
        cropped = canvas[y:y+h, x:x+w]
        
        if cropped.size == 0:
            return None
        
        # Resize to target size
        resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((3, 3), np.uint8)
        resized = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel)
        resized = cv2.morphologyEx(resized, cv2.MORPH_OPEN, kernel)
        
        return resized
    
    def _get_all_strokes_bounding_box(self) -> Optional[Tuple[int, int, int, int]]:
        """Get the bounding box that encompasses all strokes."""
        if not self.strokes:
            return None
        
        all_points = []
        for stroke in self.strokes:
            all_points.extend(stroke.points)
        
        if not all_points:
            return None
        
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def has_strokes(self) -> bool:
        """Check if there are any strokes in the drawing."""
        return len(self.strokes) > 0
    
    def clear_drawing(self):
        """Clear all strokes and reset the drawing state."""
        self.strokes.clear()
        self.current_stroke = None
        self.point_history.clear()
        self.drawing_canvas = np.zeros((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8)
    
    def get_stroke_count(self) -> int:
        """Get the number of completed strokes."""
        return len(self.strokes)
    
    def get_total_points(self) -> int:
        """Get the total number of points in all strokes."""
        total = sum(len(stroke.points) for stroke in self.strokes)
        if self.current_stroke:
            total += len(self.current_stroke.points)
        return total
    
    def get_drawing_stats(self) -> Dict[str, int]:
        """Get statistics about the current drawing."""
        return {
            'stroke_count': self.get_stroke_count(),
            'total_points': self.get_total_points(),
            'has_current_stroke': self.current_stroke is not None,
            'current_stroke_points': len(self.current_stroke.points) if self.current_stroke else 0
        }
