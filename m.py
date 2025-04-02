import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

NUM_HANDS = 2
NUM_POSE_JOINTS = 6
NUM_HAND_KEYPOINTS = 21
FEATURES_PER_FRAME = 1260
REFERENCE_IMAGE_SIZE = (640, 480)

LANDMARK_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
CONNECTION_STYLE = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)

def extract_arm_features(results_pose, results_hands):
    features = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
    offset = 0
    
    arm_data = {
        "left": {"coords": np.zeros(9), "velocity": np.zeros(9), "acceleration": np.zeros(9)},
        "right": {"coords": np.zeros(9), "velocity": np.zeros(9), "acceleration": np.zeros(9)}
    }

    if results_pose.pose_landmarks:
        pose_landmarks = results_pose.pose_landmarks.landmark
        arm_joints = {
            'left': [mp_pose.PoseLandmark.LEFT_SHOULDER,
                    mp_pose.PoseLandmark.LEFT_ELBOW,
                    mp_pose.PoseLandmark.LEFT_WRIST],
            'right': [mp_pose.PoseLandmark.RIGHT_SHOULDER,
                     mp_pose.PoseLandmark.RIGHT_ELBOW,
                     mp_pose.PoseLandmark.RIGHT_WRIST]
        }
        
        for side in ['left', 'right']:
            arm_points = []
            for joint in arm_joints[side]:
                lm = pose_landmarks[joint]
                arm_points.extend([
                    lm.x * REFERENCE_IMAGE_SIZE[0],
                    lm.y * REFERENCE_IMAGE_SIZE[1],
                    lm.z * REFERENCE_IMAGE_SIZE[0]
                ])
            
            arm_data[side]["coords"] = np.array(arm_points, dtype=np.float32)

    hand_features = np.zeros(900, dtype=np.float32)
    if results_hands.multi_hand_landmarks:
        hands = {
            "left": np.zeros(NUM_HAND_KEYPOINTS*3),
            "right": np.zeros(NUM_HAND_KEYPOINTS*3)
        }
        
        for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            handedness = results_hands.multi_handedness[idx].classification[0].label.lower()
            keypoints = np.array([
                (lm.x * REFERENCE_IMAGE_SIZE[0], 
                 lm.y * REFERENCE_IMAGE_SIZE[1], 
                 lm.z * REFERENCE_IMAGE_SIZE[0])
                for lm in hand_landmarks.landmark
            ]).flatten()
            
            hands[handedness][:len(keypoints)] = keypoints

        for hand in ["left", "right"]:
            wrist = hands[hand][:3]
            if np.any(wrist != 0):
                hands[hand] = hands[hand].reshape(-1, 3) - wrist
                hands[hand] = hands[hand].flatten()
            
            hand_features[offset:offset+63] = hands[hand][:63]
            offset += 63

        hand_geo_features = calculate_hand_geometrics(hands)
        hand_features[offset:offset+len(hand_geo_features)] = hand_geo_features
        offset += len(hand_geo_features)

    arm_geo_features = calculate_arm_geometrics(arm_data)
    features[:len(arm_geo_features)] = arm_geo_features
    features[len(arm_geo_features):len(arm_geo_features)+900] = hand_features
    
    return features

def calculate_arm_geometrics(arm_data):
    features = []
    
    for side in ['left', 'right']:
        joints = arm_data[side]["coords"].reshape(-1, 3)
        
        if np.any(joints != 0):
            v1 = joints[1] - joints[0]
            v2 = joints[2] - joints[1]
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
            features.extend([np.sin(angle), np.cos(angle), angle])  # Added raw angle
        else:
            features.extend([0, 0, 0])
        
        if np.any(joints != 0):
            upper_arm = np.linalg.norm(joints[1] - joints[0])
            lower_arm = np.linalg.norm(joints[2] - joints[1])
            total_arm = np.linalg.norm(joints[2] - joints[0])  # Added total length
            features.extend([upper_arm / (lower_arm + 1e-6), total_arm])
        else:
            features.extend([0, 0])
    
    return np.array(features, dtype=np.float32)

def calculate_hand_geometrics(hands):
    features = []
    
    for hand in ["left", "right"]:
        keypoints = hands[hand].reshape(-1, 3)
        
        if np.all(keypoints == 0):
            features.extend([0]*50)  # Adjusted size
            continue
            
        finger_connections = {'thumb': [1, 4], 'index': [5, 8], 
                             'middle': [9, 12], 'ring': [13, 16], 'pinky': [17, 20]}
        for finger in finger_connections.values():
            base = keypoints[finger[0]]
            tip = keypoints[finger[1]]
            features.extend(tip - base)
        
        palm_center = np.mean(keypoints[[0, 5, 9, 13, 17]], axis=0)
        features.extend(palm_center)
        
        for combo in [(4, 8), (8, 12), (12, 16), (16, 20)]:
            v1 = keypoints[combo[0]] - palm_center
            v2 = keypoints[combo[1]] - palm_center
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            features.append(cos_angle)
            features.append(np.linalg.norm(v1 - v2))  # Added distance
    
    return np.array(features, dtype=np.float32)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    try:
        results = model.process(image)
    except Exception as e:
        print(f"MediaPipe Error: {e}")
        results = None
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

def visualize_landmarks(image, results_pose, results_hands):
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=LANDMARK_STYLE,
            connection_drawing_spec=CONNECTION_STYLE
        )
    
    if results_hands.multi_hand_landmarks:
        for landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=LANDMARK_STYLE,
                connection_drawing_spec=CONNECTION_STYLE
            )
    return image