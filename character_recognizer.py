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
    
    def __init__(self, use_ocr: bool = True, use_ml: bool = False, debug: bool = False):
        """
        Initialize the character recognizer.
        
        Args:
            use_ocr: Whether to use OCR for character recognition
            use_ml: Whether to use machine learning model (requires training)
            debug: Whether to save debug images
        """
        self.use_ocr = use_ocr
        self.use_ml = use_ml
        self.debug = False  # force off to avoid writing files
        self.debug_counter = 0
        
        # Initialize OCR reader with better settings
        if self.use_ocr:
            try:
                # Use more languages and better settings for hand-drawn text
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context
                self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                print("OCR reader initialized successfully")
            except Exception as e:
                print(f"Failed to initialize OCR reader: {e}")
                print("Falling back to template matching only")
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
        
        # Check if image has reasonable content
        if processed_image.mean() < 5:  # Too dark
            print("Recognition: Image too dark, skipping")
            return None
        
        # Try different recognition methods
        recognized_char = None
        
        # Method 1: OCR recognition (primary method)
        if self.use_ocr:
            recognized_char = self._recognize_with_ocr(processed_image)
            if recognized_char and self._is_valid_character(recognized_char):
                print(f"OCR recognized: {recognized_char}")
                return recognized_char
        
        # Method 2: ML model recognition
        if self.use_ml and self.ml_ready:
            recognized_char = self._recognize_with_ml(processed_image)
            if recognized_char and self._is_valid_character(recognized_char):
                print(f"ML recognized: {recognized_char}")
                return recognized_char
        
        # Method 3: Template matching (very conservative fallback)
        # Only use template matching if OCR completely fails
        print("OCR failed, trying conservative template matching...")
        recognized_char = self._recognize_with_templates_conservative(processed_image)
        if recognized_char and self._is_valid_character(recognized_char):
            print(f"Template recognized: {recognized_char}")
            return recognized_char
        
        print("No character recognized")
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
        
        # Save debug image
        # debug image saving disabled
        
        # Apply gentle Gaussian blur to reduce noise without losing details
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Use adaptive thresholding for better results with varying lighting
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Apply gentle morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Resize to larger size for better recognition (256x256 instead of 64x64)
        resized = cv2.resize(cleaned, (256, 256), interpolation=cv2.INTER_CUBIC)
        
        # Save debug image
        # debug image saving disabled
        
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
            # Use larger image size for better OCR results
            ocr_image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
            
            # Apply additional preprocessing for OCR
            ocr_image = cv2.bitwise_not(ocr_image)  # Invert for OCR
            
            # Add padding around the character for better recognition
            padded_image = cv2.copyMakeBorder(ocr_image, 20, 20, 20, 20, 
                                            cv2.BORDER_CONSTANT, value=0)
            
            # Save debug image
            # debug image saving disabled
            
            # Run OCR with better parameters
            results = self.ocr_reader.readtext(padded_image, 
                                            detail=1, 
                                            paragraph=False,
                                            width_ths=0.7,
                                            height_ths=0.7,
                                            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-+')
            
            if results and len(results) > 0:
                # Get the result with highest confidence
                best_result = max(results, key=lambda x: x[2])  # x[2] is confidence
                recognized_text = str(best_result[1]).strip()
                confidence = best_result[2]
                
                print(f"OCR result: '{recognized_text}' (confidence: {confidence:.2f})")
                
                # Return first character if multiple characters detected
                if len(recognized_text) > 0 and confidence > 0.3:  # Lower confidence threshold
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
        
        # Save debug image
        # debug image saving disabled
        
        # Check if image has enough content to be a valid character
        if image.mean() < 10:  # Too dark/empty
            print("Template matching: Image too dark, skipping")
            return None
        
        # Get image properties for validation
        image_area = image.shape[0] * image.shape[1]
        white_pixels = np.sum(image > 127)
        white_ratio = white_pixels / image_area
        
        print(f"Template matching - Image stats: mean={image.mean():.1f}, white_ratio={white_ratio:.3f}")
        
        # Analyze image shape characteristics
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape_info = "unknown"
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.8:
                    shape_info = "circular"
                elif circularity > 0.5:
                    shape_info = "oval"
                else:
                    shape_info = "angular"
        
        print(f"Shape analysis: {shape_info}")
        
        for char, template in self.templates.items():
            # Resize template to match image size
            template_resized = cv2.resize(template, image.shape[::-1])
            
            # Try multiple matching methods
            methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
            max_scores = []
            
            for method in methods:
                correlation = cv2.matchTemplate(image, template_resized, method)
                _, max_val, _, _ = cv2.minMaxLoc(correlation)
                max_scores.append(max_val)
            
            # Use the best score from all methods
            score = max(max_scores)
            
            # Additional validation for specific characters
            if char == '0':  # Circle - be very strict
                if score > best_score and score > 0.6:  # Much higher threshold for circle
                    # Additional check: image should be roughly circular
                    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(largest_contour)
                        perimeter = cv2.arcLength(largest_contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.7:  # Must be fairly circular
                                best_score = score
                                best_match = char
            elif char in ['1', 'I']:  # Lines - check if image is mostly vertical
                if score > best_score and score > 0.4:
                    # Check if image has strong vertical structure
                    vertical_projection = np.sum(image, axis=1)
                    horizontal_projection = np.sum(image, axis=0)
                    if len(vertical_projection) > 0 and len(horizontal_projection) > 0:
                        max_vertical = np.max(vertical_projection)
                        max_horizontal = np.max(horizontal_projection)
                        # Vertical should be much stronger than horizontal
                        if max_vertical > max_horizontal * 1.5 and max_vertical > image.shape[1] * 0.4:
                            best_score = score
                            best_match = char
            else:  # Other characters
                if score > best_score and score > 0.35:  # Higher threshold
                    best_score = score
                    best_match = char
        
        print(f"Template matching - Best match: {best_match}, Score: {best_score:.3f}")
        
        # Return match if score is above threshold and image has reasonable content
        if best_score > 0.35 and white_ratio > 0.02:  # At least 2% white pixels, higher threshold
            return best_match
        
        return None
    
    def _recognize_with_templates_conservative(self, image: np.ndarray) -> Optional[str]:
        """
        Very conservative template matching that only matches very clear cases.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Recognized character or None
        """
        # Only try a few very distinctive templates
        simple_templates = {
            '0': self._create_circle_template((64, 64)),
            '1': self._create_line_template((64, 64), vertical=True),
            'I': self._create_line_template((64, 64), vertical=True),
            'O': self._create_circle_template((64, 64)),
        }
        
        best_match = None
        best_score = 0
        
        for char, template in simple_templates.items():
            # Resize template to match image size
            template_resized = cv2.resize(template, image.shape[::-1])
            
            # Use only the most reliable matching method
            correlation = cv2.matchTemplate(image, template_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(correlation)
            
            # Very high threshold for conservative matching
            if max_val > 0.7 and max_val > best_score:
                best_score = max_val
                best_match = char
        
        print(f"Conservative template matching - Best match: {best_match}, Score: {best_score:.3f}")
        
        # Only return if very confident
        if best_score > 0.7:
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
            '6': self._create_six_template(template_size),
            '7': self._create_seven_template(template_size),
            '8': self._create_eight_template(template_size),
            '9': self._create_nine_template(template_size),
        }
        
        # Letter templates (very basic)
        letters = {
            'A': self._create_a_template(template_size),
            'B': self._create_b_template(template_size),
            'C': self._create_c_template(template_size),
            'D': self._create_d_template(template_size),
            'E': self._create_e_template(template_size),
            'F': self._create_f_template(template_size),
            'G': self._create_g_template(template_size),
            'H': self._create_h_template(template_size),
            'I': self._create_line_template(template_size, vertical=True),
            'J': self._create_j_template(template_size),
            'K': self._create_k_template(template_size),
            'L': self._create_l_template(template_size),
            'M': self._create_m_template(template_size),
            'N': self._create_n_template(template_size),
            'O': self._create_circle_template(template_size),
            'P': self._create_p_template(template_size),
            'Q': self._create_q_template(template_size),
            'R': self._create_r_template(template_size),
            'S': self._create_s_template(template_size),
            'T': self._create_t_template(template_size),
            'U': self._create_u_template(template_size),
            'V': self._create_v_template(template_size),
            'W': self._create_w_template(template_size),
            'X': self._create_x_template(template_size),
            'Y': self._create_y_template(template_size),
            'Z': self._create_z_template(template_size),
        }
        
        self.templates.update(digits)
        self.templates.update(letters)
    
    def _create_circle_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a circle template."""
        template = np.zeros(size, dtype=np.uint8)
        center = (size[1] // 2, size[0] // 2)
        radius = min(size) // 3
        # Create a thicker circle for better matching
        cv2.circle(template, center, radius, 255, 4)
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
    
    # Additional template creation methods
    def _create_a_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create an A template."""
        template = np.zeros(size, dtype=np.uint8)
        # Left diagonal
        cv2.line(template, (size[1] // 4, 3 * size[0] // 4), (size[1] // 2, size[0] // 4), 255, 2)
        # Right diagonal
        cv2.line(template, (size[1] // 2, size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Horizontal crossbar
        cv2.line(template, (size[1] // 3, size[0] // 2), (2 * size[1] // 3, size[0] // 2), 255, 2)
        return template
    
    def _create_b_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a B template."""
        template = np.zeros(size, dtype=np.uint8)
        # Left vertical line
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 4, 3 * size[0] // 4), 255, 2)
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
    
    def _create_c_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a C template."""
        template = np.zeros(size, dtype=np.uint8)
        # Left vertical line
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Top horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 4), (3 * size[1] // 4, size[0] // 4), 255, 2)
        # Bottom horizontal line
        cv2.line(template, (size[1] // 4, 3 * size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        return template
    
    def _create_d_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a D template."""
        template = np.zeros(size, dtype=np.uint8)
        # Left vertical line
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Top horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 4), (3 * size[1] // 4, size[0] // 4), 255, 2)
        # Bottom horizontal line
        cv2.line(template, (size[1] // 4, 3 * size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Right vertical line
        cv2.line(template, (3 * size[1] // 4, size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        return template
    
    def _create_e_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create an E template."""
        template = np.zeros(size, dtype=np.uint8)
        # Left vertical line
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Top horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 4), (3 * size[1] // 4, size[0] // 4), 255, 2)
        # Middle horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 2), (3 * size[1] // 4, size[0] // 2), 255, 2)
        # Bottom horizontal line
        cv2.line(template, (size[1] // 4, 3 * size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        return template
    
    def _create_f_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create an F template."""
        template = np.zeros(size, dtype=np.uint8)
        # Left vertical line
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Top horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 4), (3 * size[1] // 4, size[0] // 4), 255, 2)
        # Middle horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 2), (3 * size[1] // 4, size[0] // 2), 255, 2)
        return template
    
    def _create_g_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a G template."""
        template = np.zeros(size, dtype=np.uint8)
        # Left vertical line
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Top horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 4), (3 * size[1] // 4, size[0] // 4), 255, 2)
        # Bottom horizontal line
        cv2.line(template, (size[1] // 4, 3 * size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Right vertical line (partial)
        cv2.line(template, (3 * size[1] // 4, size[0] // 2), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Right horizontal line
        cv2.line(template, (size[1] // 2, 3 * size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        return template
    
    def _create_j_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a J template."""
        template = np.zeros(size, dtype=np.uint8)
        # Top horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 4), (3 * size[1] // 4, size[0] // 4), 255, 2)
        # Right vertical line
        cv2.line(template, (3 * size[1] // 4, size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Bottom horizontal line
        cv2.line(template, (size[1] // 4, 3 * size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Left vertical line (partial)
        cv2.line(template, (size[1] // 4, 2 * size[0] // 3), (size[1] // 4, 3 * size[0] // 4), 255, 2)
        return template
    
    def _create_k_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a K template."""
        template = np.zeros(size, dtype=np.uint8)
        # Left vertical line
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Top diagonal
        cv2.line(template, (size[1] // 4, size[0] // 2), (3 * size[1] // 4, size[0] // 4), 255, 2)
        # Bottom diagonal
        cv2.line(template, (size[1] // 4, size[0] // 2), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        return template
    
    def _create_m_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create an M template."""
        template = np.zeros(size, dtype=np.uint8)
        # Left vertical line
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Right vertical line
        cv2.line(template, (3 * size[1] // 4, size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Left diagonal
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 2, size[0] // 2), 255, 2)
        # Right diagonal
        cv2.line(template, (size[1] // 2, size[0] // 2), (3 * size[1] // 4, size[0] // 4), 255, 2)
        return template
    
    def _create_n_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create an N template."""
        template = np.zeros(size, dtype=np.uint8)
        # Left vertical line
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Right vertical line
        cv2.line(template, (3 * size[1] // 4, size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Diagonal
        cv2.line(template, (size[1] // 4, size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        return template
    
    def _create_p_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a P template."""
        template = np.zeros(size, dtype=np.uint8)
        # Left vertical line
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Top horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 4), (3 * size[1] // 4, size[0] // 4), 255, 2)
        # Middle horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 2), (3 * size[1] // 4, size[0] // 2), 255, 2)
        # Right vertical line (partial)
        cv2.line(template, (3 * size[1] // 4, size[0] // 4), (3 * size[1] // 4, size[0] // 2), 255, 2)
        return template
    
    def _create_q_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a Q template."""
        template = np.zeros(size, dtype=np.uint8)
        # Circle
        center = (size[1] // 2, size[0] // 2)
        radius = min(size) // 3
        cv2.circle(template, center, radius, 255, 2)
        # Tail
        cv2.line(template, (size[1] // 2, 2 * size[0] // 3), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        return template
    
    def _create_r_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create an R template."""
        template = np.zeros(size, dtype=np.uint8)
        # Left vertical line
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Top horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 4), (3 * size[1] // 4, size[0] // 4), 255, 2)
        # Middle horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 2), (3 * size[1] // 4, size[0] // 2), 255, 2)
        # Right vertical line (partial)
        cv2.line(template, (3 * size[1] // 4, size[0] // 4), (3 * size[1] // 4, size[0] // 2), 255, 2)
        # Diagonal
        cv2.line(template, (size[1] // 4, size[0] // 2), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        return template
    
    def _create_v_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a V template."""
        template = np.zeros(size, dtype=np.uint8)
        # Left diagonal
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 2, 3 * size[0] // 4), 255, 2)
        # Right diagonal
        cv2.line(template, (size[1] // 2, 3 * size[0] // 4), (3 * size[1] // 4, size[0] // 4), 255, 2)
        return template
    
    def _create_w_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a W template."""
        template = np.zeros(size, dtype=np.uint8)
        # Left vertical line
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Right vertical line
        cv2.line(template, (3 * size[1] // 4, size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Left diagonal
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 2, 3 * size[0] // 4), 255, 2)
        # Right diagonal
        cv2.line(template, (size[1] // 2, 3 * size[0] // 4), (3 * size[1] // 4, size[0] // 4), 255, 2)
        return template
    
    def _create_x_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create an X template."""
        template = np.zeros(size, dtype=np.uint8)
        # Diagonal 1
        cv2.line(template, (size[1] // 4, size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Diagonal 2
        cv2.line(template, (3 * size[1] // 4, size[0] // 4), (size[1] // 4, 3 * size[0] // 4), 255, 2)
        return template
    
    def _create_y_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a Y template."""
        template = np.zeros(size, dtype=np.uint8)
        # Left diagonal
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 2, size[0] // 2), 255, 2)
        # Right diagonal
        cv2.line(template, (size[1] // 2, size[0] // 2), (3 * size[1] // 4, size[0] // 4), 255, 2)
        # Vertical line
        cv2.line(template, (size[1] // 2, size[0] // 2), (size[1] // 2, 3 * size[0] // 4), 255, 2)
        return template
    
    def _create_z_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a Z template."""
        template = np.zeros(size, dtype=np.uint8)
        # Top horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 4), (3 * size[1] // 4, size[0] // 4), 255, 2)
        # Diagonal
        cv2.line(template, (3 * size[1] // 4, size[0] // 4), (size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Bottom horizontal line
        cv2.line(template, (size[1] // 4, 3 * size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        return template
    
    # Additional digit templates
    def _create_six_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a 6 template."""
        template = np.zeros(size, dtype=np.uint8)
        # Left vertical line
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Top horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 4), (3 * size[1] // 4, size[0] // 4), 255, 2)
        # Middle horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 2), (3 * size[1] // 4, size[0] // 2), 255, 2)
        # Bottom horizontal line
        cv2.line(template, (size[1] // 4, 3 * size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Right vertical line (partial)
        cv2.line(template, (3 * size[1] // 4, size[0] // 2), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        return template
    
    def _create_seven_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a 7 template."""
        template = np.zeros(size, dtype=np.uint8)
        # Top horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 4), (3 * size[1] // 4, size[0] // 4), 255, 2)
        # Diagonal
        cv2.line(template, (3 * size[1] // 4, size[0] // 4), (size[1] // 4, 3 * size[0] // 4), 255, 2)
        return template
    
    def _create_eight_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create an 8 template."""
        template = np.zeros(size, dtype=np.uint8)
        # Left vertical line
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Right vertical line
        cv2.line(template, (3 * size[1] // 4, size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Top horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 4), (3 * size[1] // 4, size[0] // 4), 255, 2)
        # Middle horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 2), (3 * size[1] // 4, size[0] // 2), 255, 2)
        # Bottom horizontal line
        cv2.line(template, (size[1] // 4, 3 * size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        return template
    
    def _create_nine_template(self, size: Tuple[int, int]) -> np.ndarray:
        """Create a 9 template."""
        template = np.zeros(size, dtype=np.uint8)
        # Left vertical line (partial)
        cv2.line(template, (size[1] // 4, size[0] // 4), (size[1] // 4, size[0] // 2), 255, 2)
        # Right vertical line
        cv2.line(template, (3 * size[1] // 4, size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        # Top horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 4), (3 * size[1] // 4, size[0] // 4), 255, 2)
        # Middle horizontal line
        cv2.line(template, (size[1] // 4, size[0] // 2), (3 * size[1] // 4, size[0] // 2), 255, 2)
        # Bottom horizontal line
        cv2.line(template, (size[1] // 4, 3 * size[0] // 4), (3 * size[1] // 4, 3 * size[0] // 4), 255, 2)
        return template
