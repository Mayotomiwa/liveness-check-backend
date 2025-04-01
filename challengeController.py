import cv2
import numpy as np
from flask import Flask, request, jsonify
import face_recognition
import random
import base64
import io
from PIL import Image

app = Flask(__name__)

class LivenessDetector:
    def __init__(self):
        self.challenge_types = ["blink", "nod", "turn_right", "turn_left"]
        self.current_challenge = None
        self.previous_landmarks = None
        self.challenge_completed = False
        
    def generate_challenge(self):
        """Generate a random liveness challenge"""
        self.current_challenge = random.choice(self.challenge_types)
        self.challenge_completed = False
        return self.current_challenge
    
    def detect_face(self, image_data):
        """Detect if there's a face in the image"""
        face_locations = face_recognition.face_locations(image_data)
        if not face_locations:
            return False, "No face detected"
        
        # Only allow one face
        if len(face_locations) > 1:
            return False, "Multiple faces detected"
            
        return True, face_locations[0]
    
    def detect_blink(self, image_data):
        """Detect if the person blinked"""
        face_landmarks = face_recognition.face_landmarks(image_data)
        if not face_landmarks:
            return False, "No face landmarks detected"
            
        # Get eye landmarks
        left_eye = face_landmarks[0]["left_eye"]
        right_eye = face_landmarks[0]["right_eye"]
        
        # Calculate the eye aspect ratio (EAR)
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        
        # If EAR is below threshold, eyes are closed (blink detected)
        if left_ear < 0.2 and right_ear < 0.2:
            return True, "Blink detected"
            
        return False, "No blink detected"
    
    def detect_head_movement(self, image_data, movement_type):
        """Detect head nodding or turning"""
        face_landmarks = face_recognition.face_landmarks(image_data)
        if not face_landmarks:
            return False, "No face landmarks detected"
            
        # If no previous landmarks, store current and return
        if self.previous_landmarks is None:
            self.previous_landmarks = face_landmarks[0]
            return False, "Initial position captured"
            
        current_landmarks = face_landmarks[0]
        
        if movement_type == "nod":
            # Check vertical movement (y-coordinate changes)
            prev_y = sum(point[1] for point in self.previous_landmarks["nose_tip"]) / len(self.previous_landmarks["nose_tip"])
            curr_y = sum(point[1] for point in current_landmarks["nose_tip"]) / len(current_landmarks["nose_tip"])
            
            if abs(curr_y - prev_y) > 10:
                self.previous_landmarks = None  # Reset
                return True, "Nod detected"
                
        elif movement_type in ["turn_right", "turn_left"]:
            # Check horizontal movement (x-coordinate changes)
            prev_x = sum(point[0] for point in self.previous_landmarks["nose_tip"]) / len(self.previous_landmarks["nose_tip"])
            curr_x = sum(point[0] for point in current_landmarks["nose_tip"]) / len(current_landmarks["nose_tip"])
            
            if movement_type == "turn_right" and (curr_x - prev_x) > 10:
                self.previous_landmarks = None  # Reset
                return True, "Right turn detected"
            elif movement_type == "turn_left" and (prev_x - curr_x) > 10:
                self.previous_landmarks = None  # Reset
                return True, "Left turn detected"
        
        # Update landmarks for next comparison
        self.previous_landmarks = current_landmarks
        return False, "No movement detected"
    
    def _calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio for blink detection"""
        # Calculate the vertical distances
        v1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        v2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        
        # Calculate the horizontal distance
        h = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        
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

@app.route('/api/challenge', methods=['GET'])
def get_challenge():
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
    
    # Decode the base64 image
    try:
        image_data = base64.b64decode(request.json['image'])
        image = Image.open(io.BytesIO(image_data))
        # Convert to format face_recognition can use
        np_image = np.array(image)
        # Convert RGB to BGR (if needed)
        if len(np_image.shape) == 3 and np_image.shape[2] == 3:
            np_image = np_image[:, :, ::-1]
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