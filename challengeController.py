import cv2
import numpy as np
from flask import Flask, request, jsonify
import random
import base64
import io
from PIL import Image
import mediapipe as mp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

class LivenessDetector:
    def __init__(self):
        self.challenge_types = ["blink", "nod", "turn_right", "turn_left"]
        self.current_challenge = None
        self.previous_landmarks = None
        self.challenge_completed = False
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
            
        # Define landmarks indices for specific facial features
        # MediaPipe face mesh has 468 landmarks
        # These indices are for key points we need
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Key points around left eye
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Key points around right eye
        self.NOSE_TIP_INDEX = 1  # Nose tip
    
    def generate_challenge(self):
        """Generate a random liveness challenge"""
        self.current_challenge = random.choice(self.challenge_types)
        self.challenge_completed = False
        return self.current_challenge
    
    def detect_face(self, image_data):
        """Detect if there's a face in the image using MediaPipe"""
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        
        # Process the image with MediaPipe
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return False, "No face detected"
        
        # Only allow one face
        if len(results.multi_face_landmarks) > 1:
            return False, "Multiple faces detected"
            
        # Return the face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get image dimensions for normalization
        height, width = image_data.shape[:2]
        
        # Get bounding box of face (approximate)
        x_min = width
        y_min = height
        x_max = 0
        y_max = 0
        
        for landmark in face_landmarks.landmark:
            x, y = int(landmark.x * width), int(landmark.y * height)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)
        
        # Return the face location as (x, y, w, h)
        return True, (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def get_facial_landmarks(self, image_data):
        """Extract facial landmarks using MediaPipe"""
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        
        # Process the image with MediaPipe
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get image dimensions for normalization
        height, width = image_data.shape[:2]
        
        # Extract landmarks
        landmarks = {}
        
        # Get coordinates for eyes
        landmarks["left_eye"] = []
        landmarks["right_eye"] = []
        landmarks["nose_tip"] = []
        
        # Extract face landmarks
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # Get left eye landmarks
        for idx in self.LEFT_EYE_INDICES:
            landmarks["left_eye"].append(
                (int(face_landmarks[idx].x * width), 
                 int(face_landmarks[idx].y * height))
            )
        
        # Get right eye landmarks
        for idx in self.RIGHT_EYE_INDICES:
            landmarks["right_eye"].append(
                (int(face_landmarks[idx].x * width), 
                 int(face_landmarks[idx].y * height))
            )
        
        # Get nose tip landmark
        landmarks["nose_tip"].append(
            (int(face_landmarks[self.NOSE_TIP_INDEX].x * width), 
             int(face_landmarks[self.NOSE_TIP_INDEX].y * height))
        )
        
        return landmarks
    
    def detect_blink(self, image_data):
        """Detect if the person blinked using MediaPipe landmarks"""
        landmarks = self.get_facial_landmarks(image_data)
        
        if not landmarks or "left_eye" not in landmarks or "right_eye" not in landmarks:
            return False, "Could not detect eyes"
        
        # Calculate Eye Aspect Ratio (EAR) for both eyes
        left_ear = self._calculate_ear(landmarks["left_eye"])
        right_ear = self._calculate_ear(landmarks["right_eye"])
        
        # If EAR is below threshold, eyes are closed (blink detected)
        # This threshold may need tuning based on testing
        threshold = 0.2
        if left_ear < threshold and right_ear < threshold:
            return True, "Blink detected (low EAR)"
            
        return False, "No blink detected"
    
    def detect_head_movement(self, image_data, movement_type):
        """Detect head nodding or turning using MediaPipe landmarks"""
        landmarks = self.get_facial_landmarks(image_data)
        
        if not landmarks or "nose_tip" not in landmarks:
            return False, "No facial landmarks detected"
        
        # If no previous landmarks, store current and return
        if self.previous_landmarks is None:
            self.previous_landmarks = landmarks
            return False, "Initial position captured"
        
        if movement_type == "nod":
            # Check vertical movement (y-coordinate changes)
            prev_y = sum(point[1] for point in self.previous_landmarks["nose_tip"]) / len(self.previous_landmarks["nose_tip"])
            curr_y = sum(point[1] for point in landmarks["nose_tip"]) / len(landmarks["nose_tip"])
            
            # Threshold for detecting vertical movement
            if abs(curr_y - prev_y) > 15:  # Adjusted threshold for MediaPipe
                self.previous_landmarks = None  # Reset
                return True, "Nod detected"
                
        elif movement_type in ["turn_right", "turn_left"]:
            # Check horizontal movement (x-coordinate changes)
            prev_x = sum(point[0] for point in self.previous_landmarks["nose_tip"]) / len(self.previous_landmarks["nose_tip"])
            curr_x = sum(point[0] for point in landmarks["nose_tip"]) / len(landmarks["nose_tip"])
            
            # Threshold for detecting horizontal movement
            if movement_type == "turn_right" and (curr_x - prev_x) > 15:  # Adjusted threshold
                self.previous_landmarks = None  # Reset
                return True, "Right turn detected"
            elif movement_type == "turn_left" and (prev_x - curr_x) > 15:  # Adjusted threshold
                self.previous_landmarks = None  # Reset
                return True, "Left turn detected"
        
        # Update landmarks for next comparison
        self.previous_landmarks = landmarks
        return False, "No movement detected"
    
    def _calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio for blink detection"""
        # Need at least 6 points for traditional EAR calculation
        if len(eye_points) < 6:
            return 0.3  # Default value, not indicating a blink
            
        # Calculate the vertical distances
        v1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        v2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        
        # Calculate the horizontal distance
        h = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        
        # Avoid division by zero
        if h == 0:
            return 0.3
            
        # Calculate EAR
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def verify_challenge(self, image_data):
        """Verify if the current challenge was completed successfully"""
        if not self.current_challenge:
            return False, "No active challenge"
            
        if self.current_challenge == "blink":
            result, message = self.detect_blink(image_data)
        elif self.current_challenge == "nod":
            result, message = self.detect_head_movement(image_data, "nod")
        elif self.current_challenge == "turn_right":
            result, message = self.detect_head_movement(image_data, "turn_right")
        elif self.current_challenge == "turn_left":
            result, message = self.detect_head_movement(image_data, "turn_left")
            
        if result:
            self.challenge_completed = True
            
        return result, message

# Create detector instance
detector = LivenessDetector()

@app.route('/api/start-session', methods=['GET'])
def start_session():
    """Start a new liveness check session"""
    session_id = random.randint(10000, 99999)
    return jsonify({
        'success': True,
        'session_id': str(session_id),
        'message': 'Session started'
    })

@app.route('/api/challenge/<session_id>', methods=['GET'])
def get_challenge(session_id):
    """Generate a new liveness challenge"""
    challenge = detector.generate_challenge()
    return jsonify({
        'challenge': challenge,
        'instructions': get_challenge_instructions(challenge)
    })

@app.route('/api/verify', methods=['POST'])
def verify_liveness():
    """Verify a liveness challenge from an image"""
    if 'image' not in request.json:
        return jsonify({'success': False, 'message': 'No image provided'})
    
    session_id = request.json.get('session_id')
    challenge = request.json.get('challenge')
    
    if not session_id or not challenge:
        return jsonify({'success': False, 'message': 'Missing session_id or challenge'})
    
    # Decode the base64 image
    try:
        image_data = base64.b64decode(request.json['image'])
        image = Image.open(io.BytesIO(image_data))
        # Convert to format OpenCV can use
        np_image = np.array(image)
        # Convert RGB to BGR if needed
        if len(np_image.shape) == 3 and np_image.shape[2] == 3:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error decoding image: {str(e)}'})
    
    # First detect if there's a face
    face_detected, face_result = detector.detect_face(np_image)
    if not face_detected:
        return jsonify({'success': False, 'message': face_result})
    
    # Verify the challenge
    success, message = detector.verify_challenge(np_image)
    
    return jsonify({
        'success': success,
        'message': message,
        'challenge_completed': detector.challenge_completed
    })

def get_challenge_instructions(challenge):
    """Return user-friendly instructions for each challenge type"""
    instructions = {
        "blink": "Please blink your eyes",
        "nod": "Please nod your head up and down",
        "turn_right": "Please turn your head to the right",
        "turn_left": "Please turn your head to the left"
    }
    return instructions.get(challenge, "Follow the instructions on screen")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)