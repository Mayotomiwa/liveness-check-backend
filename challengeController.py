import cv2
import numpy as np
import base64
import io
from PIL import Image
import mediapipe as mp
import os
import random
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("liveness-detector")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize FastAPI app
app = FastAPI()

# Store active sessions
active_sessions = {}

class LivenessDetector:
    def __init__(self):
        self.challenge_types = ["blink", "nod", "turn_right", "turn_left"]
        self.current_challenges = []  # Store multiple challenges
        self.challenge_index = 0
        self.previous_landmarks = None
        self.completed_challenges = set()
        self.final_verification_complete = False
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
            
        # Define landmarks indices for specific facial features
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.NOSE_TIP_INDEX = 1
        
        # Store initial face position for the final verification
        self.initial_face_position = None
        self.movement_history = []
        
    def generate_challenges(self, num_challenges=3):
        """Generate a sequence of random liveness challenges"""
        # Make sure we don't repeat challenges
        self.current_challenges = random.sample(self.challenge_types, min(num_challenges, len(self.challenge_types)))
        self.challenge_index = 0
        self.completed_challenges = set()
        self.final_verification_complete = False
        return self.current_challenges
    
    def get_current_challenge(self):
        """Get the current active challenge"""
        if self.challenge_index < len(self.current_challenges):
            return self.current_challenges[self.challenge_index]
        return "stay_still"  # Final verification step
    
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
            
        # Get image dimensions for normalization
        height, width = image_data.shape[:2]
        
        # Get bounding box of face (approximate)
        x_min = width
        y_min = height
        x_max = 0
        y_max = 0
        
        for landmark in results.multi_face_landmarks[0].landmark:
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
        
        # Extract face landmarks
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # Get left eye landmarks
        landmarks["left_eye"] = []
        for idx in self.LEFT_EYE_INDICES:
            landmarks["left_eye"].append(
                (int(face_landmarks[idx].x * width), 
                 int(face_landmarks[idx].y * height))
            )
        
        # Get right eye landmarks
        landmarks["right_eye"] = []
        for idx in self.RIGHT_EYE_INDICES:
            landmarks["right_eye"].append(
                (int(face_landmarks[idx].x * width), 
                 int(face_landmarks[idx].y * height))
            )
        
        # Get nose tip landmark
        landmarks["nose_tip"] = []
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
        threshold = 0.2
        if left_ear < threshold and right_ear < threshold:
            return True, "Blink detected"
            
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
            if abs(curr_y - prev_y) > 15:
                self.previous_landmarks = None  # Reset
                return True, "Nod detected"
                
        elif movement_type in ["turn_right", "turn_left"]:
            # Check horizontal movement (x-coordinate changes)
            prev_x = sum(point[0] for point in self.previous_landmarks["nose_tip"]) / len(self.previous_landmarks["nose_tip"])
            curr_x = sum(point[0] for point in landmarks["nose_tip"]) / len(landmarks["nose_tip"])
            
            # Threshold for detecting horizontal movement
            if movement_type == "turn_right" and (curr_x - prev_x) > 15:
                self.previous_landmarks = None  # Reset
                return True, "Right turn detected"
            elif movement_type == "turn_left" and (prev_x - curr_x) > 15:
                self.previous_landmarks = None  # Reset
                return True, "Left turn detected"
        
        # Update landmarks for next comparison
        self.previous_landmarks = landmarks
        return False, "No movement detected"
    
    def verify_still(self, image_data):
        """Verify the user is staying still for the final verification step"""
        landmarks = self.get_facial_landmarks(image_data)
        
        if not landmarks or "nose_tip" not in landmarks:
            return False, "No facial landmarks detected"
            
        # Initialize the reference position if not set
        if self.initial_face_position is None:
            self.initial_face_position = landmarks
            self.movement_history = []
            return False, "Initial position captured for verification"
            
        # Get current nose position
        curr_nose_x = sum(point[0] for point in landmarks["nose_tip"]) / len(landmarks["nose_tip"])
        curr_nose_y = sum(point[1] for point in landmarks["nose_tip"]) / len(landmarks["nose_tip"])
        
        # Get reference nose position
        ref_nose_x = sum(point[0] for point in self.initial_face_position["nose_tip"]) / len(self.initial_face_position["nose_tip"])
        ref_nose_y = sum(point[1] for point in self.initial_face_position["nose_tip"]) / len(self.initial_face_position["nose_tip"])
        
        # Calculate movement amount
        movement = np.sqrt((curr_nose_x - ref_nose_x)**2 + (curr_nose_y - ref_nose_y)**2)
        
        # Keep track of movement history for stability analysis
        self.movement_history.append(movement)
        if len(self.movement_history) > 30:  # Track last 30 frames
            self.movement_history.pop(0)
        
        # Check if movement is below threshold (stable position)
        threshold = 10  # Threshold for considering movement insignificant
        if len(self.movement_history) >= 15 and all(m < threshold for m in self.movement_history[-15:]):
            self.final_verification_complete = True
            return True, "Verification complete - user is staying still"
            
        return False, "Continue staying still"
    
    def _calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio for blink detection"""
        if len(eye_points) < 6:
            return 0.3
            
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
        current_challenge = self.get_current_challenge()
        
        # If all challenges are completed, do the final verification
        if current_challenge == "stay_still":
            return self.verify_still(image_data)
            
        # Otherwise verify the current active challenge
        if current_challenge == "blink":
            result, message = self.detect_blink(image_data)
        elif current_challenge == "nod":
            result, message = self.detect_head_movement(image_data, "nod")
        elif current_challenge == "turn_right":
            result, message = self.detect_head_movement(image_data, "turn_right")
        elif current_challenge == "turn_left":
            result, message = self.detect_head_movement(image_data, "turn_left")
        else:
            return False, "Unknown challenge"
            
        if result:
            # Mark this challenge as completed
            self.completed_challenges.add(current_challenge)
            # Move to the next challenge if this one is completed
            self.challenge_index += 1
            # Reset for the next challenge
            self.previous_landmarks = None
            
        return result, message
        
    def is_liveness_verified(self):
        """Check if all challenges have been completed and final verification is done"""
        return (len(self.completed_challenges) == len(self.current_challenges) and 
                self.final_verification_complete)

def get_challenge_instructions(challenge):
    """Return user-friendly instructions for each challenge type"""
    instructions = {
        "blink": "Please blink your eyes",
        "nod": "Please nod your head up and down",
        "turn_right": "Please turn your head to the right",
        "turn_left": "Please turn your head to the left",
        "stay_still": "Please stay still for final verification"
    }
    return instructions.get(challenge, "Follow the instructions on screen")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = None
    detector = None
    
    try:
        # Wait for initialization message
        data = await websocket.receive_text()
        init_data = json.loads(data)
        
        if init_data.get('type') == 'init':
            # Verify API key if needed
            if init_data.get('apiKey') != 'liveness_detection_key_2025':
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'message': 'Invalid API key'
                }))
                return
            
            # Initialize session
            session_id = f"session_{random.randint(10000, 99999)}"
            detector = LivenessDetector()
            challenges = detector.generate_challenges(3)
            active_sessions[session_id] = detector
            
            # Send session info
            await websocket.send_text(json.dumps({
                'type': 'session',
                'sessionId': session_id,
                'challenges': challenges,
                'currentChallenge': detector.get_current_challenge(),
                'instructions': get_challenge_instructions(detector.get_current_challenge())
            }))
            
            # Process frames
            while True:
                data = await websocket.receive_text()
                frame_data = json.loads(data)
                
                if frame_data.get('type') != 'frame':
                    continue
                
                # Decode image
                try:
                    image_data = base64.b64decode(frame_data['image'])
                    image = Image.open(io.BytesIO(image_data))
                    np_image = np.array(image)
                    if len(np_image.shape) == 3 and np_image.shape[2] == 3:
                        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.error(f"Image decode error: {str(e)}")
                    await websocket.send_text(json.dumps({
                        'type': 'error',
                        'message': 'Invalid image data'
                    }))
                    continue
                
                # Process frame
                face_detected, face_result = detector.detect_face(np_image)
                if not face_detected:
                    await websocket.send_text(json.dumps({
                        'type': 'verification',
                        'success': False,
                        'message': face_result
                    }))
                    continue
                
                success, message = detector.verify_challenge(np_image)
                response = {
                    'type': 'verification',
                    'success': success,
                    'message': message,
                    'currentChallenge': detector.get_current_challenge(),
                    'instructions': get_challenge_instructions(detector.get_current_challenge()),
                    'challengesCompleted': list(detector.completed_challenges),
                    'challengesTotal': len(detector.current_challenges),
                    'allCompleted': detector.is_liveness_verified()
                }
                await websocket.send_text(json.dumps(response))
                
                if detector.is_liveness_verified():
                    await websocket.send_text(json.dumps({
                        'type': 'complete',
                        'success': True,
                        'message': "Liveness verification complete"
                    }))
                    break
    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        await websocket.send_text(json.dumps({
            'type': 'error',
            'message': f"Server error: {str(e)}"
        }))
    finally:
        if session_id in active_sessions:
            del active_sessions[session_id]

async def main():
    import uvicorn
    config = uvicorn.Config(
        "challengeController:app",
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()