import cv2
import numpy as np
import base64
import io
import time
from PIL import Image
import mediapipe as mp
import os
import random
import json
import logging
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from uvicorn import Config, Server
from uvicorn.lifespan.on import LifespanOn
from uvicorn.protocols.websockets.auto import AutoWebSocketsProtocol
from uvicorn.protocols.http.auto import AutoHTTPProtocol

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("liveness-detector")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants - Updated values
CHALLENGE_TIMEOUT = 45  # Reduced from 50 to 45 seconds
PING_INTERVAL = 10      # Reduced from 15 to 10 seconds
CONNECTION_TIMEOUT = 60  # Overall connection timeout
MAX_MULTIPLE_FACE_DETECTIONS = 3
VERIFICATION_FAILED_FLAG = "verification_failed"
SESSION_TIMEOUT = 300    # 5 minutes for session timeout

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
  mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
            
        # Define landmarks indices for specific facial features
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.NOSE_TIP_INDEX = 1
        
        # Store initial face position for the final verification
        self.initial_face_position = None
        self.movement_history = []
        self.stable_frame_count = 0
        
        # Face verification counters
        self.multiple_face_counter = 0
        self.verification_failed = False
        
        # For action verification
        self.last_detected_action = None
        self.action_detection_counter = 0
        self.consecutive_action_threshold = 2  # Reduced from 3 to 2
        
        # For blink detection
        self.ear_history = []
        
        # For turn detection
        self.position_history = []
        
    def generate_challenges(self, num_challenges=3):
        """Generate a sequence of random liveness challenges"""
        # Make sure we don't repeat challenges
        self.current_challenges = random.sample(self.challenge_types, min(num_challenges, len(self.challenge_types)))
        self.challenge_index = 0
        self.completed_challenges = set()
        self.final_verification_complete = False
        self.verification_failed = False
        self.multiple_face_counter = 0
        self.ear_history = []
        self.position_history = []
        self.movement_history = []
        self.stable_frame_count = 0
        return self.current_challenges
    
    def get_current_challenge(self):
        """Get the current active challenge"""
        if self.verification_failed:
            return VERIFICATION_FAILED_FLAG
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
            self.multiple_face_counter += 1
            # If multiple faces detected too many times, fail verification permanently
            if self.multiple_face_counter >= MAX_MULTIPLE_FACE_DETECTIONS:
                self.verification_failed = True
                return False, "Verification failed: Multiple faces detected repeatedly"
            return False, f"Multiple faces detected ({self.multiple_face_counter}/{MAX_MULTIPLE_FACE_DETECTIONS})"
        else:
            # Reset counter if only one face is detected
            self.multiple_face_counter = 0
            
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
    
    # Lower threshold makes it easier to detect blinks
    threshold = 0.2  # Lowered from 0.23
    
    # Store historical EAR values to detect an actual blink sequence
    avg_ear = (left_ear + right_ear) / 2
    self.ear_history.append(avg_ear)
    if len(self.ear_history) > 10:  # Keep fewer frames for faster detection
        self.ear_history.pop(0)
    
    # Much more lenient blink detection
    blink_detected = False
    if len(self.ear_history) >= 3:  # Reduced from 4 to 3 frames
        # Simpler pattern detection: was open, then closed
        if (max(self.ear_history[:-2]) > threshold and  # Was open earlier
            min(self.ear_history[-2:]) < threshold * 1.2):  # Closed with higher tolerance
            blink_detected = True
    
    # Detect blink with faster confirmation
    if blink_detected:
        # Lower the bar for consistent detection
        if self.last_detected_action == "blink":
            self.action_detection_counter += 1.5  # Increase counter faster
        else:
            self.last_detected_action = "blink"
            self.action_detection_counter = 1
            
        # Only need to see the action once clearly
        self.consecutive_action_threshold = 1  # Reduced from 2
        
        # Return true once we've reached the threshold
        if self.action_detection_counter >= self.consecutive_action_threshold:
            return True, "Blink detected"
    else:
        # Slower decay for intermittent detection
        if self.last_detected_action == "blink" and self.action_detection_counter > 0:
            self.action_detection_counter = max(0, self.action_detection_counter - 0.3)  # Reduced from 0.5
            
    return False, "No blink detected"

def detect_head_movement(self, image_data, movement_type):
    """Detect head nodding or turning using MediaPipe landmarks"""
    landmarks = self.get_facial_landmarks(image_data)
    
    if not landmarks or "nose_tip" not in landmarks:
        return False, "No facial landmarks detected"
    
    # Get nose position
    curr_x = sum(point[0] for point in landmarks["nose_tip"]) / len(landmarks["nose_tip"])
    curr_y = sum(point[1] for point in landmarks["nose_tip"]) / len(landmarks["nose_tip"])
    
    # Store position history
    self.position_history.append((curr_x, curr_y))
    if len(self.position_history) > 15:  # Reduced from 20 for faster detection
        self.position_history.pop(0)
    
    # If we don't have enough history yet, just return
    if len(self.position_history) < 3:  # Reduced from 5 for faster detection
        self.previous_landmarks = landmarks
        return False, "Collecting initial position data"
    
    action_detected = False
    message = "No movement detected"
    detected_action = None
    
    # For turn detection, analyze the movement sequence
    if movement_type in ["turn_right", "turn_left"]:
        # Calculate x positions (fewer frames)
        x_positions = [pos[0] for pos in self.position_history[-8:]]  # Reduced from 10
        
        # For right turn: need rightward movement (increasing x)
        if movement_type == "turn_right":
            # Reduced threshold from 15 to 10 pixels
            if (x_positions[-1] - x_positions[0]) > 10:  # More sensitive rightward movement detection
                # Much more tolerant consistency check
                consistent_count = sum(1 for i in range(len(x_positions)-1) if x_positions[i] <= x_positions[i+1] + 10)
                if consistent_count >= len(x_positions) * 0.6:  # Only require 60% consistency (was 70%)
                    action_detected = True
                    detected_action = "turn_right"
                    message = "Right turn detected"
        
        # For left turn: need leftward movement (decreasing x)
        elif movement_type == "turn_left":
            # Reduced threshold from 15 to 10 pixels
            if (x_positions[0] - x_positions[-1]) > 10:  # More sensitive leftward movement detection
                # Much more tolerant consistency check
                consistent_count = sum(1 for i in range(len(x_positions)-1) if x_positions[i] >= x_positions[i+1] - 10)
                if consistent_count >= len(x_positions) * 0.6:  # Only require 60% consistency (was 70%)
                    action_detected = True
                    detected_action = "turn_left"
                    message = "Left turn detected"
    
    # Handle nodding with similar improvements
    elif movement_type == "nod":
        # Calculate y positions
        y_positions = [pos[1] for pos in self.position_history[-8:]]  # Fewer frames
        # Reduced threshold from 12 to 8
        y_range = max(y_positions) - min(y_positions)
        if y_range > 8:  # Much more sensitive
            action_detected = True
            detected_action = "nod"
            message = "Nod detected"
    
    # Check if we're detecting the action
    if detected_action == movement_type:
        if self.last_detected_action == detected_action:
            self.action_detection_counter += 1.5  # Increase counter faster
        else:
            self.last_detected_action = detected_action
            self.action_detection_counter = 1
        
        # Need fewer confirmations
        self.consecutive_action_threshold = 1  # Reduced from 2
        
        # Return true if we've seen the action enough times
        if self.action_detection_counter >= self.consecutive_action_threshold:
            # Reset for next challenge
            self.position_history = []
            self.previous_landmarks = None
            return True, message
    else:
        # Slower decay
        if self.last_detected_action == movement_type and self.action_detection_counter > 0:
            self.action_detection_counter = max(0, self.action_detection_counter - 0.3)  # Reduced from 0.5
    
    self.previous_landmarks = landmarks
    return False, "Continue " + get_challenge_instructions(movement_type).lower()
    def verify_still(self, image_data):
        """Verify the user is staying still for the final verification step"""
        landmarks = self.get_facial_landmarks(image_data)
        
        if not landmarks or "nose_tip" not in landmarks:
            return False, "No facial landmarks detected"
        
        # Extract current nose position
        curr_nose_x = sum(point[0] for point in landmarks["nose_tip"]) / len(landmarks["nose_tip"])
        curr_nose_y = sum(point[1] for point in landmarks["nose_tip"]) / len(landmarks["nose_tip"])
        
        # Always reset if we're just starting the verification or if substantial movement occurred
        if self.initial_face_position is None or self.movement_history is None:
            self.initial_face_position = landmarks
            self.movement_history = []
            self.stable_frame_count = 0
            return False, "Initial position captured for verification"
        
        # Get reference nose position
        ref_nose_x = sum(point[0] for point in self.initial_face_position["nose_tip"]) / len(self.initial_face_position["nose_tip"])
        ref_nose_y = sum(point[1] for point in self.initial_face_position["nose_tip"]) / len(self.initial_face_position["nose_tip"])
        
        # Calculate movement amount
        movement = np.sqrt((curr_nose_x - ref_nose_x)**2 + (curr_nose_y - ref_nose_y)**2)
        
        # Keep track of movement history for stability analysis
        self.movement_history.append(movement)
        if len(self.movement_history) > 15:  # Track last 15 frames
            self.movement_history.pop(0)
        
        # If large movement detected, reset the reference position
        if movement > 25:  # Large movement threshold
            self.initial_face_position = landmarks
            self.movement_history = []
            self.stable_frame_count = 0
            return False, "Movement detected. Please stay still."
        
        # Check if movement is below threshold (stable position)
        threshold = 10  # Threshold for considering movement insignificant
        
        # Count consecutive stable frames
        if movement < threshold:
            self.stable_frame_count = getattr(self, 'stable_frame_count', 0) + 1
        else:
            self.stable_frame_count = 0
        
        # Need 15 consecutive stable frames
        if self.stable_frame_count >= 15:
            self.final_verification_complete = True
            return True, "Verification complete - user is staying still"
        
        return False, f"Continue staying still ({self.stable_frame_count}/15 frames)"
    
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
        
        # If verification has failed permanently, don't process further
        if current_challenge == VERIFICATION_FAILED_FLAG:
            return False, "Verification failed due to security concerns"
            
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
            # Reset action detection counters
            self.last_detected_action = None
            self.action_detection_counter = 0
            
            # Mark this challenge as completed
            self.completed_challenges.add(current_challenge)
            # Move to the next challenge if this one is completed
            self.challenge_index += 1
            # Reset for the next challenge
            self.previous_landmarks = None
            self.ear_history = []
            self.position_history = []
            
        return result, message
        
    def is_liveness_verified(self):
        """Check if all challenges have been completed and final verification is done"""
        if self.verification_failed:
            return False
        return (len(self.completed_challenges) == len(self.current_challenges) and 
                self.final_verification_complete)


def get_challenge_instructions(challenge):
    """Return user-friendly instructions for each challenge type"""
    instructions = {
        "blink": "Please blink your eyes",
        "nod": "Please nod your head up and down",
        "turn_right": "Please turn your head to the right",
        "turn_left": "Please turn your head to the left",
        "stay_still": "Please stay still for final verification",
        VERIFICATION_FAILED_FLAG: "Verification failed. Please restart the process."
    }
    return instructions.get(challenge, "Follow the instructions on screen")

if init_data.get('sessionId') in active_sessions:
    saved_session = active_sessions[init_data.get('sessionId')]
    detector = saved_session['detector']
    detector.challenge_index = saved_session['challenge_index']
    detector.ear_history = saved_session['ear_history']
    detector.position_history = saved_session['position_history']
    detector.last_detected_action = saved_session['last_detected_action']
    detector.action_detection_counter = saved_session['action_detection_counter']


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = None
    detector = None
    last_activity = time.time()
    last_pong = time.time()  # Track last pong time
    ping_task = None
    challenge_timer = None
    session_timer = None

    async def send_ping():
        """Regularly send ping messages to keep connection alive"""
        while True:
            await asyncio.sleep(PING_INTERVAL)
            try:
                if websocket.client_state == 1:  # 1 = CONNECTED
                    await websocket.send_text(json.dumps({'type': 'ping'}))
                    logger.debug(f"Sent ping to {session_id}")
            except Exception as e:
                logger.error(f"Ping error: {str(e)}")
                break

    async def challenge_timeout_handler():
        """Handle challenge timeout"""
        try:
            await asyncio.sleep(CHALLENGE_TIMEOUT)
            if websocket.client_state == 1:  # 1 = CONNECTED
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'message': f'Challenge timed out after {CHALLENGE_TIMEOUT} seconds'
                }))
                await websocket.close(code=1008)
        except asyncio.CancelledError:
            pass

    async def session_timeout_handler():
        """Handle overall session timeout"""
        try:
            await asyncio.sleep(SESSION_TIMEOUT)
            if websocket.client_state == 1:  # 1 = CONNECTED
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'message': f'Session timed out after {SESSION_TIMEOUT} seconds'
                }))
                await websocket.close(code=1008)
        except asyncio.CancelledError:
            pass

    try:
        # Initial handshake with timeout
# Reduce initial handshake timeout
try:
    data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)  # 10.0 to 5.0            init_data = json.loads(data)
            last_activity = time.time()
            last_pong = time.time()
            
            if init_data.get('type') == 'ping':
                await websocket.send_text(json.dumps({'type': 'pong'}))
                return
                
            if init_data.get('type') == 'init':
                # API Key verification
                if init_data.get('apiKey') != 'liveness_detection_key_2025':
                    await websocket.send_text(json.dumps({
                        'type': 'error',
                        'message': 'Invalid API key'
                    }))
                    await websocket.close(code=4001)
                    return
                
                # Initialize session
                session_id = f"session_{random.randint(10000, 99999)}"
                detector = LivenessDetector()
                challenges = detector.generate_challenges(3)
                active_sessions[session_id] = {
                    'detector': detector,
                    'last_active': time.time(),
                    'challenges': challenges,
                    'completed': []
                }
                
                # Start ping task
                ping_task = asyncio.create_task(send_ping())
                # Start session timer
                session_timer = asyncio.create_task(session_timeout_handler())
                
                # Send session info
                await websocket.send_text(json.dumps({
                    'type': 'session',
                    'sessionId': session_id,
                    'challenges': challenges,
                    'currentChallenge': detector.get_current_challenge(),
                    'instructions': get_challenge_instructions(detector.get_current_challenge()),
                    'timeout': CHALLENGE_TIMEOUT,
                    'sessionTimeout': SESSION_TIMEOUT
                }))
                
                # Main processing loop
                while True:
                    try:
                        # Check overall connection timeout
                        if time.time() - last_activity > CONNECTION_TIMEOUT:
                            logger.info(f"Connection timeout for {session_id}")
                            await websocket.close(code=1008)
                            break
                            
                        # Check pong response
                        if time.time() - last_pong > PING_INTERVAL * 2:
                            logger.warning(f"No pong received for {session_id}")
                            await websocket.close(code=4003)
                            break
                            
                        # Receive data with timeout
                        data = await asyncio.wait_for(
                            websocket.receive_text(),
                            timeout=min(PING_INTERVAL, CHALLENGE_TIMEOUT)
                        )
                        last_activity = time.time()
                        message = json.loads(data)
                        active_sessions[session_id]['last_active'] = time.time()
                        
                        # Handle ping/pong
                        if message.get('type') == 'ping':
                            await websocket.send_text(json.dumps({'type': 'pong'}))
                            continue
                        elif message.get('type') == 'pong':
                            last_pong = time.time()
                            continue
                            
                        # Process frame messages
                        if message.get('type') == 'frame':
                            # Reset challenge timer
                            if challenge_timer:
                                challenge_timer.cancel()
                            challenge_timer = asyncio.create_task(challenge_timeout_handler())
                            
                            # Process the frame
                            try:
                                image_data = base64.b64decode(message['image'])
                                image = Image.open(io.BytesIO(image_data))
                                np_image = np.array(image)
                                if len(np_image.shape) == 3 and np_image.shape[2] == 3:
                                    np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
                                    
                                # Verification logic
                                face_detected, face_result = detector.detect_face(np_image)
                                
                                # Handle failed verification permanently
                                if detector.verification_failed:
                                    await websocket.send_text(json.dumps({
                                        'type': 'verification',
                                        'success': False,
                                        'message': "Verification failed due to security concerns",
                                        'currentChallenge': VERIFICATION_FAILED_FLAG,
                                        'instructions': get_challenge_instructions(VERIFICATION_FAILED_FLAG),
                                        'challengesCompleted': list(detector.completed_challenges),
                                        'challengesTotal': len(detector.current_challenges),
                                        'allCompleted': False,
                                        'verificationFailed': True
                                    }))
                                    # End the session after a permanent failure
                                    await websocket.close(code=4002)
                                    break
                                
                                if not face_detected:
                                    await websocket.send_text(json.dumps({
                                        'type': 'verification',
                                        'success': False,
                                        'message': face_result,
                                        'currentChallenge': detector.get_current_challenge(),
                                        'instructions': get_challenge_instructions(detector.get_current_challenge()),
                                        'challengesCompleted': list(detector.completed_challenges),
                                        'challengesTotal': len(detector.current_challenges)
                                    }))
                                    continue
                                    
                                success, message_text = detector.verify_challenge(np_image)
                                response = {
                                    'type': 'verification',
                                    'success': success,
                                    'message': message_text,
                                    'currentChallenge': detector.get_current_challenge(),
                                    'instructions': get_challenge_instructions(detector.get_current_challenge()),
                                    'challengesCompleted': list(detector.completed_challenges),
                                    'challengesTotal': len(detector.current_challenges),
                                    'allCompleted': detector.is_liveness_verified(),
                                    'timeRemaining': CHALLENGE_TIMEOUT - (time.time() - last_activity),
                                    'sessionTimeRemaining': SESSION_TIMEOUT - (time.time() - last_activity)
                                }
                                await websocket.send_text(json.dumps(response))
                                
                                if detector.is_liveness_verified():
                                    await websocket.send_text(json.dumps({
                                        'type': 'complete',
                                        'success': True,
                                        'message': "Liveness verification complete",
                                        'sessionId': session_id
                                    }))
                                    break
                                    
                            except Exception as e:
                                logger.error(f"Frame processing error: {str(e)}")
                                await websocket.send_text(json.dumps({
                                    'type': 'error',
                                    'message': 'Frame processing failed'
                                }))
                                
                    except asyncio.TimeoutError:
                        # No data received within timeout period
                        continue
                        
        except asyncio.TimeoutError:
            logger.info("Initial handshake timeout")
            await websocket.close(code=1008)
            return
            
    except WebSocketDisconnect as e:
        logger.info(f"Client disconnected: {session_id} (code: {e.code})")
        # Save session state for possible reconnection
        if session_id and detector:
            active_sessions[session_id] = {
                'detector': detector,
                'last_active': time.time(),
                'challenges': detector.current_challenges,
                'completed': list(detector.completed_challenges)
            }
    except json.JSONDecodeError:
        logger.error("Invalid JSON received")
        await websocket.close(code=4000)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        try:
            await websocket.send_text(json.dumps({
                'type': 'error',
                'message': f"Server error: {str(e)}"
            }))
        except:
            pass
    finally:
        # Clean up resources
        if challenge_timer and not challenge_timer.done():
            challenge_timer.cancel()
        if session_timer and not session_timer.done():
            session_timer.cancel()
        if ping_task and not ping_task.done():
            ping_task.cancel()
        try:
            await websocket.close()
        except:
            pass

@app.get("/session/{session_id}")
async def get_session_state(session_id: str):
    """Endpoint to check session state for reconnection"""
    if session_id in active_sessions:
        session = active_sessions[session_id]
        return {
            "status": "active",
            "sessionId": session_id,
            "challenges": session['challenges'],
            "completed": session['completed'],
            "last_active": time.time() - session['last_active']
        }
    return {"status": "not_found"}

async def main():
    config = Config(
        "challengeController:app",
        host="0.0.0.0",
        port=10000,
        log_level="info",
        timeout_keep_alive=60,
        ws_ping_interval=PING_INTERVAL,
        ws_ping_timeout=30,
        lifespan="on",
        reload=False
    )
    server = Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())