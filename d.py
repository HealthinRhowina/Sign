import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from scipy.interpolate import interp1d
from m import mediapipe_detection, extract_arm_features  # Explicit import from m.py

# Configuration
DATA_PATH = 'zdata1'
VIDEO_PATH = "sl"
ACTIONS = [
    'again', 'all', 'also', 'always', 'angry', 'answer', 'any', 'ask', 
    'bad', 'bathroom', 'beautiful', 'bed', 'big', 'bird', 'book', 
    'bread', 'buy', 'call', 'car', 'cat', 'chair', 'child', 'clean', 
    'clothes', 'close', 'cold', 'color', 'come', 'computer', 'cook', 'coffee', 
    'different', 'doctor', 'dog', 'door', 'drink', 'early', 'eat', 'easy', 
    'evening', 'family', 'few', 'feel', 'find', 'fish', 'first', 'food', 
    'free', 'friend', 'fruit', 'fun', 'game', 'give', 'go', 'good', 'happy', 
    'hard', 'hat', 'have', 'health', 'hear', 'help', 'her', 'here',
    'his', 'home', 'hospital', 'hot', 'house', 'hour', 'how', 'hungry','I', 
    'idea', 'important','know', 'last', 'late', 
    'learn', 'leave', 'less', 'like', 'love', 'make', 'many', 
    'maybe', 'me', 'medicine', 'meet', 'milk', 'minute', 'money', 'month', 
    'more', 'morning', 'movie', 'music', 'my', 'name', 'need', 'new', 'next', 
    'nice', 'night', 'no', 'now', 'number', 'often', 'old', 'one', 'open', 
    'our', 'paper','phone', 'please', 'poor', 'price', 'problem', 
    'question', 'rain', 'read', 'ready', 'real', 'rich', 'right', 'run', 
    'sad', 'safe', 'same', 'say', 'school', 'see', 'she', 'shirt', 
    'shop', 'short', 'should', 'sick', 'small', 'snow', 'some', 'sometimes', 
    'song', 'start', 'stay', 'stop', 'store', 'street', 'strong', 'student', 
    'sun', 'sure', 'take', 'talk', 'tea', 'teach', 'teacher', 'their', 
    'them', 'there', 'they', 'think', 'this', 'three', 
    'time', 'tired', 'together', 'tomorrow', 'tree', 'true', 'two', 
    'understand','vegetable', 'very', 'wait', 'walk', 'want', 'warm', 
    'wash', 'water', 'we', 'weak', 'weather', 'week', 'what', 
    'when', 'where', 'which', 'who', 'why', 'window', 'woman', 'work', 
    'write', 'wrong', 'year', 'yes', 'yesterday', 'young', 'your'
]
SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 1260
MIN_FRAMES = 15

def temporal_augmentation(sequence):
    """Enhanced temporal augmentation with more variation"""
    original_length = len(sequence)
    x = np.linspace(0, 1, original_length)
    new_length = int(original_length * np.random.uniform(0.7, 1.3))  # Wider range
    new_x = np.linspace(0, 1, new_length)
    
    interpolator = interp1d(x, sequence, axis=0, kind='cubic')  # Smoother interpolation
    return interpolator(new_x)

def compute_motion_features(sequence):
    """Enhanced motion features with jerk calculation"""
    sequence = np.array(sequence, dtype=np.float32)
    
    velocity = np.diff(sequence, axis=0, prepend=sequence[[0]])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[[0]])
    jerk = np.diff(acceleration, axis=0, prepend=acceleration[[0]])  # Added jerk
    
    combined = np.concatenate([sequence, velocity, acceleration, jerk], axis=1)
    
    if combined.shape[0] < SEQUENCE_LENGTH:
        pad_size = SEQUENCE_LENGTH - combined.shape[0]
        combined = np.pad(combined, [(0, pad_size), (0, 0)], mode='reflect')
    
    return combined[:SEQUENCE_LENGTH]

def process_video(video_path, pose, hands):
    cap = cv2.VideoCapture(video_path)
    frames = []
    last_valid = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        _, results_pose = mediapipe_detection(frame, pose)
        _, results_hands = mediapipe_detection(frame, hands)
        
        features = extract_arm_features(results_pose, results_hands)
        
        if features is not None and np.any(features):
            last_valid = features
            frames.append(features)
        elif last_valid is not None:
            frames.append(last_valid)
    
    cap.release()
    
    if len(frames) < MIN_FRAMES:
        return None
    
    if np.random.rand() < 0.7:  # Increase augmentation frequency
        frames = temporal_augmentation(frames)
    
    return frames

def process_action(action, pose, hands):
    action_path = os.path.join(VIDEO_PATH, action)
    save_path = os.path.join(DATA_PATH, action)
    os.makedirs(save_path, exist_ok=True)

    video_files = sorted([f for f in os.listdir(action_path) if f.endswith('.mp4')])
    
    for idx, video_file in enumerate(video_files):
        video_path = os.path.join(action_path, video_file)
        
        try:
            frames = process_video(video_path, pose, hands)
            if frames is None:
                print(f"⚠ Skipping short video: {video_file}")
                continue

            motion_features = compute_motion_features(frames)
            
            seq_path = os.path.join(save_path, str(idx))
            os.makedirs(seq_path, exist_ok=True)
            
            for frame_num in range(SEQUENCE_LENGTH):
                np.save(
                    os.path.join(seq_path, f'{frame_num}.npy'),
                    motion_features[frame_num]
                )
                
        except Exception as e:
            print(f"🚨 Error processing {video_file}: {str(e)}")

def main():
    os.makedirs (DATA_PATH, exist_ok=True)
    
    with mp.solutions.pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.8,  # Higher confidence
        min_tracking_confidence=0.8
    ) as pose, mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    ) as hands:
        
        for action in tqdm(ACTIONS, desc="Processing Actions"):
            action_path = os.path.join(VIDEO_PATH, action)
            if not os.path.exists(action_path):
                print(f"⚠ Missing action directory: {action}")
                continue
            process_action(action, pose, hands)

if __name__ == "__main__":
    main()
    print("✅ Arm-hand data processing complete with enhanced motion features.")