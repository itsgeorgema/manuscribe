"""
Hand tracking module using MediaPipe for gesture recognition.
"""

import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class HandTracker:
    def __init__(self, 
                 static_image_mode: bool = False,
                 max_num_hands: int = 1,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the hand tracker.
        
        Args:
            static_image_mode: Whether to treat the input images as static
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Hand landmark indices
        self.FINGER_TIP_IDS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        self.FINGER_PIP_IDS = [3, 6, 10, 14, 18]  # PIP joints for finger extension detection
        self.FINGER_MCP_IDS = [2, 5, 9, 13, 17]   # MCP joints
        
    def detect_hands(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect hands in the given image.
        
        Args:
            image: RGB image array
            
        Returns:
            List of detected hands with landmarks
        """
        results = self.hands.process(image)
        detected_hands = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_info = {
                    'landmarks': hand_landmarks,
                    'landmark_list': [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                }
                detected_hands.append(hand_info)
        
        return detected_hands
    
    def get_index_finger_tip(self, hand: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """
        Get the index finger tip position.
        
        Args:
            hand: Hand information dictionary
            
        Returns:
            (x, y) coordinates of index finger tip (normalized 0-1)
        """
        if 'landmark_list' in hand and len(hand['landmark_list']) > 8:
            # Index finger tip is landmark 8
            landmark = hand['landmark_list'][8]
            return (landmark[0], landmark[1])
        return None
    
    def get_all_finger_tips(self, hand: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """
        Get all finger tip positions.
        
        Args:
            hand: Hand information dictionary
            
        Returns:
            Dictionary with finger names as keys and (x, y) coordinates as values
        """
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        finger_tips = {}
        
        if 'landmark_list' in hand:
            for i, tip_id in enumerate(self.FINGER_TIP_IDS):
                if len(hand['landmark_list']) > tip_id:
                    landmark = hand['landmark_list'][tip_id]
                    finger_tips[finger_names[i]] = (landmark[0], landmark[1])
        
        return finger_tips
    
    def is_finger_extended(self, hand: Dict[str, Any], finger_name: str = 'index') -> bool:
        """
        Check if a specific finger is extended.
        
        Args:
            hand: Hand information dictionary
            finger_name: Name of the finger to check ('thumb', 'index', 'middle', 'ring', 'pinky')
            
        Returns:
            True if finger is extended, False otherwise
        """
        finger_map = {
            'thumb': 0,
            'index': 1,
            'middle': 2,
            'ring': 3,
            'pinky': 4
        }
        
        if finger_name not in finger_map:
            return False
        
        finger_idx = finger_map[finger_name]
        
        if 'landmark_list' not in hand or len(hand['landmark_list']) <= max(self.FINGER_TIP_IDS):
            return False
        
        landmarks = hand['landmark_list']
        
        # For thumb, check x-coordinate difference
        if finger_name == 'thumb':
            tip_x = landmarks[self.FINGER_TIP_IDS[finger_idx]][0]
            mcp_x = landmarks[self.FINGER_MCP_IDS[finger_idx]][0]
            return abs(tip_x - mcp_x) > 0.04
        else:
            # For other fingers, check y-coordinate (tip should be higher than pip joint)
            tip_y = landmarks[self.FINGER_TIP_IDS[finger_idx]][1]
            pip_y = landmarks[self.FINGER_PIP_IDS[finger_idx]][1]
            return tip_y < pip_y
    
    def get_extended_fingers(self, hand: Dict[str, Any]) -> List[str]:
        """
        Get list of extended fingers.
        
        Args:
            hand: Hand information dictionary
            
        Returns:
            List of extended finger names
        """
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        extended_fingers = []
        
        for finger in finger_names:
            if self.is_finger_extended(hand, finger):
                extended_fingers.append(finger)
        
        return extended_fingers
    
    def get_finger_distance(self, hand: Dict[str, Any], finger1: str, finger2: str) -> Optional[float]:
        """
        Calculate distance between two finger tips.
        
        Args:
            hand: Hand information dictionary
            finger1: First finger name
            finger2: Second finger name
            
        Returns:
            Distance between finger tips (normalized coordinates)
        """
        finger_tips = self.get_all_finger_tips(hand)
        
        if finger1 in finger_tips and finger2 in finger_tips:
            pos1 = finger_tips[finger1]
            pos2 = finger_tips[finger2]
            
            distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            return distance
        
        return None
    
    def draw_landmarks(self, image: np.ndarray, hand: Dict[str, Any]) -> np.ndarray:
        """
        Draw hand landmarks on the image.
        
        Args:
            image: Image to draw on
            hand: Hand information dictionary
            
        Returns:
            Image with landmarks drawn
        """
        if 'landmarks' in hand:
            self.mp_drawing.draw_landmarks(
                image, 
                hand['landmarks'], 
                self.mp_hands.HAND_CONNECTIONS
            )
        
        return image
    
    def get_gesture_info(self, hand: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive gesture information for the hand.
        
        Args:
            hand: Hand information dictionary
            
        Returns:
            Dictionary with gesture information
        """
        extended_fingers = self.get_extended_fingers(hand)
        finger_tips = self.get_all_finger_tips(hand)
        
        gesture_info = {
            'extended_fingers': extended_fingers,
            'finger_tips': finger_tips,
            'num_extended': len(extended_fingers),
            'is_pointing': len(extended_fingers) == 1 and 'index' in extended_fingers,
            'is_peace': len(extended_fingers) == 2 and 'index' in extended_fingers and 'middle' in extended_fingers,
            'is_fist': len(extended_fingers) == 0,
            'is_open_hand': len(extended_fingers) >= 4
        }
        
        return gesture_info
