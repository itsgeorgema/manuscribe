"""
Character recognition module for converting drawn strokes to text.
"""

import cv2
import numpy as np
import os
import pickle
from typing import Optional, List, Dict, Tuple
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import easyocr


class CharacterRecognizer:
    """Recognizes characters from drawn stroke images."""
    
    def __init__(self, use_ocr: bool = True, use_ml: bool = False):
        """
        Initialize the character recognizer.
        
        Args:
            use_ocr: Whether to use OCR for character recognition
            use_ml: Whether to use machine learning model (requires training)
        """
        self.use_ocr = use_ocr
        self.use_ml = use_ml
        
        # Initialize OCR reader
        if self.use_ocr:
            try:
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
                print("OCR reader initialized successfully")
            except Exception as e:
                print(f"Failed to initialize OCR reader: {e}")
                self.use_ocr = False
        
        # Initialize ML components
        self.ml_model = None
        self.scaler = None
        self.ml_ready = False
        
        if self.use_ml:
            self._initialize_ml_model()
        
        # Character templates for basic pattern matching
        self.templates = {}
        self._create_basic_templates()
        
    def recognize(self, image: np.ndarray) -> Optional[str]:
        """
        Recognize a character from the input image.
        
        Args:
            image: Binary or grayscale image containing the drawn character
            
        Returns:
            Recognized character or None if recognition fails
        """
        if image is None or image.size == 0:
            return None
        
        # Preprocess the image
        processed_image = self._preprocess_image(image)
        
        # Try different recognition methods
        recognized_char = None
        
        # Method 1: OCR recognition
        if self.use_ocr:
            recognized_char = self._recognize_with_ocr(processed_image)
            if recognized_char and self._is_valid_character(recognized_char):
                return recognized_char
        
        # Method 2: ML model recognition
        if self.use_ml and self.ml_ready:
            recognized_char = self._recognize_with_ml(processed_image)
            if recognized_char and self._is_valid_character(recognized_char):
                return recognized_char
        
        # Method 3: Template matching (fallback)
        recognized_char = self._recognize_with_templates(processed_image)
        if recognized_char and self._is_valid_character(recognized_char):
            return recognized_char
        
        return None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for better recognition.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Apply binary threshold
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Resize to standard size for consistency
        resized = cv2.resize(cleaned, (64, 64), interpolation=cv2.INTER_AREA)
        
        return resized
    
    def _recognize_with_ocr(self, image: np.ndarray) -> Optional[str]:
        """
        Recognize character using OCR.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Recognized character or None
        """
        try:
            # Resize image for better OCR results
            ocr_image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)
            
            # Apply additional preprocessing for OCR
            ocr_image = cv2.bitwise_not(ocr_image)  # Invert for OCR
            
            # Run OCR
            results = self.ocr_reader.readtext(ocr_image, detail=0, paragraph=False)
            
            if results and len(results) > 0:
                # Get the first result and clean it
                recognized_text = str(results[0]).strip()
                
                # Return first character if multiple characters detected
                if len(recognized_text) > 0:
                    return recognized_text[0].upper()
            
        except Exception as e:
            print(f"OCR recognition failed: {e}")
        
        return None
    
    def _recognize_with_ml(self, image: np.ndarray) -> Optional[str]:
        """
        Recognize character using machine learning model.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Recognized character or None
        """
        if not self.ml_ready:
            return None
        
        try:
            # Extract features from image
            features = self._extract_features(image)
            features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict character
            prediction = self.ml_model.predict(features_scaled)
            confidence = max(self.ml_model.predict_proba(features_scaled)[0])
            
            # Return prediction if confidence is high enough
            if confidence > 0.7:
                return prediction[0]
            
        except Exception as e:
            print(f"ML recognition failed: {e}")
        
        return None
    
    def _recognize_with_templates(self, image: np.ndarray) -> Optional[str]:
        """
        Recognize character using template matching.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Recognized character or None
        """
        best_match = None
        best_score = 0
        
        for char, template in self.templates.items():
            # Resize template to match image size
            template_resized = cv2.resize(template, image.shape[::-1])
            
            # Calculate correlation
            correlation = cv2.matchTemplate(image, template_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(correlation)
            
            if max_val > best_score:
                best_score = max_val
                best_match = char
        
        # Return match if score is above threshold
        if best_score > 0.3:
            return best_match
        
        return None
    
    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from the image for ML recognition.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Feature vector
        """
        features = []
        
        # Basic image statistics
        features.extend([
            np.mean(image),
            np.std(image),
            np.sum(image > 127) / image.size,  # White pixel ratio
        ])
        
        # Hu moments (shape descriptors)
        moments = cv2.moments(image)
        hu_moments = cv2.HuMoments(moments).flatten()
        features.extend(hu_moments)
        
        # Contour features
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            features.extend([
                area / (image.shape[0] * image.shape[1]),  # Normalized area
                perimeter / (2 * (image.shape[0] + image.shape[1])),  # Normalized perimeter
                (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0,  # Circularity
            ])
        else:
            features.extend([0, 0, 0])
        
        # Resize image to fixed size and flatten as additional features
        flattened = cv2.resize(image, (16, 16)).flatten() / 255.0
        features.extend(flattened)
        
        return np.array(features)
    
    def _create_basic_templates(self):
        """Create basic character templates for template matching."""
        # Create simple templates for digits and common letters
        template_size = (64, 64)
        
        # Digit templates (very basic)
        digits = {
            '0': self._create_circle_template(template_size),
            '1': self._create_line_template(template_size, vertical=True),
            '2': self._create_s_template(template_size),
            '3': self._create_three_template(template_size),
            '4': self._create_four_template(template_size),
            '5': self._create_five_template(template_size),
        }
        
        # Letter templates (very basic)
        letters = {
            'H': self._create_h_template(template_size),
            'I': self._create_line_template(template_size, vertical=True),
            'L': self._create_l_template(template_size),
            'O': self._create_circle_template(template_size),
            'T': self._create_t_template(template_size),
            'U': self._create_u_template(template_size),
        }
        
        self.templates.update(digits)
        self.templates.update(letters)
    
    def _create_circle_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a circle template."""
        template = np.zeros(size, dtype=np.uint8)
        center = (size[1] // 2, size[0] // 2)
        radius = min(size) // 3
        cv2.circle(template, center, radius, 255, 2)
        return template
    
    def _create_line_template(self, size: Tuple[int, int], vertical: bool = True) -> np.ndarray:
        """Create a line template."""
        template = np.zeros(size, dtype=np.uint8)
        if vertical:
            start = (size[1] // 2, size[0] // 4)
            end = (size[1] // 2, 3 * size[0] // 4)
        else:
            start = (size[1] // 4, size[0] // 2)
            end = (3 * size[1] // 4, size[0] // 2)
        cv2.line(template, start, end, 255, 2)
        return template
    
    def _create_h_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create an H template."""
        template = np.zeros(size, dtype=np.uint8)
        # Left vertical line
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Right vertical line
        cv2.line(template, (3 * size[1] // 4, size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 2), (3 * size[1] // 4, size[0] // 2), 255, 2)
        return template
    
    def _create_l_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create an L template."""
        template = np.zeros(size, dtype=np.uint8)
        # Vertical line
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Horizontal line
        cv2.line(template, (size[1] // 4, 3 * size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        return template
    
    def _create_t_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a T template."""
        template = np.zeros(size, dtype=np.uint8)
        # Horizontal line at top
        cv2.line(template, (size[1] // 4, size[0] // 4), (3 * size[1] // 4, size[0] // 4), 255, 2)
        # Vertical line in center
        cv2.line(template, (size[1] // 2, size[0] // 4), (size[1] // 2, 3 * size[0] // 4), 255, 2)
        return template
    
    def _create_u_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a U template."""
        template = np.zeros(size, dtype=np.uint8)
        # Left vertical line
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Right vertical line
        cv2.line(template, (3 * size[1] // 4, size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Bottom horizontal line
        cv2.line(template, (size[1] // 4, 3 * size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        return template
    
    def _create_s_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create an S-like template for digit 2."""
        template = np.zeros(size, dtype=np.uint8)
        # Top horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 4), (3 * size[1] // 4, size[0] // 4), 255, 2)
        # Middle diagonal
        cv2.line(template, (3 * size[1] // 4, size[0] // 4), (size[1] // 4, size[0] // 2), 255, 2)
        # Bottom horizontal line
        cv2.line(template, (size[1] // 4, 3 * size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        return template
    
    def _create_three_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a 3 template."""
        template = np.zeros(size, dtype=np.uint8)
        # Top horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 4), (3 * size[1] // 4, size[0] // 4), 255, 2)
        # Middle horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 2), (3 * size[1] // 4, size[0] // 2), 255, 2)
        # Bottom horizontal line
        cv2.line(template, (size[1] // 4, 3 * size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Right vertical lines
        cv2.line(template, (3 * size[1] // 4, size[0] // 4), (3 * size[1] // 4, size[0] // 2), 255, 2)
        cv2.line(template, (3 * size[1] // 4, size[0] // 2), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        return template
    
    def _create_four_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a 4 template."""
        template = np.zeros(size, dtype=np.uint8)
        # Left vertical line (top half)
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 4, size[0] // 2), 255, 2)
        # Right vertical line (full)
        cv2.line(template, (3 * size[1] // 4, size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 2), (3 * size[1] // 4, size[0] // 2), 255, 2)
        return template
    
    def _create_five_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a 5 template."""
        template = np.zeros(size, dtype=np.uint8)
        # Top horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 4), (3 * size[1] // 4, size[0] // 4), 255, 2)
        # Left vertical line (top half)
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 4, size[0] // 2), 255, 2)
        # Middle horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 2), (3 * size[1] // 4, size[0] // 2), 255, 2)
        # Right vertical line (bottom half)
        cv2.line(template, (3 * size[1] // 4, size[0] // 2), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Bottom horizontal line
        cv2.line(template, (size[1] // 4, 3 * size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        return template
    
    def _initialize_ml_model(self):
        """Initialize machine learning model for character recognition."""
        # This would load a pre-trained model or initialize training
        # For now, we'll create a placeholder
        try:
            model_path = "character_model.pkl"
            scaler_path = "character_scaler.pkl"
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                with open(model_path, 'rb') as f:
                    self.ml_model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.ml_ready = True
                print("ML model loaded successfully")
            else:
                print("ML model files not found. Use template matching and OCR only.")
                self.use_ml = False
        except Exception as e:
            print(f"Failed to load ML model: {e}")
            self.use_ml = False
    
    def _is_valid_character(self, char: str) -> bool:
        """Check if the recognized character is valid."""
        if not char or len(char) == 0:
            return False
        
        # Accept alphanumeric characters and common symbols
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-+')
        return char.upper() in valid_chars
    
    def train_ml_model(self, training_data: List[Tuple[np.ndarray, str]]):
        """
        Train the ML model with labeled data.
        
        Args:
            training_data: List of (image, label) tuples
        """
        if not training_data:
            return
        
        # Extract features and labels
        features = []
        labels = []
        
        for image, label in training_data:
            processed_image = self._preprocess_image(image)
            feature_vector = self._extract_features(processed_image)
            features.append(feature_vector)
            labels.append(label.upper())
        
        features = np.array(features)
        labels = np.array(labels)
        
        # Initialize and train model
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        self.ml_model = SVC(probability=True, kernel='rbf')
        self.ml_model.fit(features_scaled, labels)
        
        # Save model
        with open("character_model.pkl", 'wb') as f:
            pickle.dump(self.ml_model, f)
        with open("character_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        self.ml_ready = True
        print(f"ML model trained with {len(training_data)} samples")
