import cv2
import numpy as np
import pyttsx3
import onnxruntime as ort
import mediapipe as mp
from collections import deque
import time
from googletrans import Translator
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
import random
from m import mediapipe_detection, extract_arm_features  # Import from m.py

nltk.download('punkt')

# Configuration adjusted
SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 1260  # Raw features from extract_arm_features
LSTM_REDUCED_FEATURES = 800  # Reduced for LSTM_CNN
RF_REDUCED_FEATURES = 1000  # Total features expected by RF model
BATCH_SIZE = 32
PREDICTION_WINDOW_SECONDS = 0.4
MIN_CONFIDENCE = 0.5
MIN_FRAMES_FOR_PREDICTION = 15
DISPLAY_DURATION = 4
WORD_DISPLAY_INTERVAL = 1

COLOR_DARK_GRAY = (50, 50, 50)
COLOR_LIGHT_GRAY = (150, 150, 150)
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 100, 0)

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

sentences = [
    "eat early",
    "stay happy",
    "teacher like tea",
    "i understand your idea",
    "i answer them",
    "please take your phone",
    "which bread your dog eat",
    "i drink coffee every morning",
    "call me tomorrow",
    "big chair small chair",
    "please give two fish",
    "She have beautiful hat",
    "see docter when sick",
    "My family eat fruit ",
    "walk home safe",
    "see your friend",
    "leave her here",
    "call your teacher",
    "beautifull house"
]

allowed_first_words = {'eat', 'stay', 'teacher', 'i', 'please', 'which', 'call', 'big', 'she', 'see', 'my', 'walk', 'leave', 'beautiful'}

tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
multi_ngram_model = defaultdict(lambda: defaultdict(int))
max_n = max(len(s) for s in tokenized_sentences)

for sentence in tokenized_sentences:
    for n in range(1, min(max_n, len(sentence) + 1)):
        for i in range(len(sentence) - n + 1):
            ngram = tuple(sentence[i:i+n-1]) if n > 1 else (sentence[i],)
            next_word = sentence[i+n-1] if i+n-1 < len(sentence) else None
            if next_word:
                multi_ngram_model[ngram][next_word] += 1

all_words = set()
for sentence in tokenized_sentences:
    all_words.update(sentence)

lstm_session = ort.InferenceSession("cnn_lstm_pca.onnx")
rf_session = ort.InferenceSession("new_rf.onnx")

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)
translator = Translator()

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)

class RealTimeFeatureCollector:
    def _init_(self):
        self.lstm_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.rf_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.timestamp_buffer = deque(maxlen=SEQUENCE_LENGTH)
        
    def process_frame(self, raw_features):
        if len(raw_features) != FEATURES_PER_FRAME:
            raise ValueError(f"Expected {FEATURES_PER_FRAME} features, got {len(raw_features)}")
            
        current_time = time.time()
        self.timestamp_buffer.append(current_time)
        
        lstm_features = self.reduce_features(raw_features, LSTM_REDUCED_FEATURES)
        self.lstm_buffer.append(lstm_features)
        self.rf_buffer.append(raw_features.copy())
        
        return self.check_prediction_readiness()
    
    def reduce_features(self, features, target_dim):
        # Placeholder - replace with actual PCA/transform from 1260 to target_dim
        return features[:target_dim]  # Temporary; use your PCA here
    
    def check_prediction_readiness(self):
        if len(self.lstm_buffer) < MIN_FRAMES_FOR_PREDICTION:
            return False, None, None
            
        lstm_sequence = np.array(self.lstm_buffer)
        rf_sequence = np.array(self.rf_buffer)
        
        if (lstm_sequence.shape != (SEQUENCE_LENGTH, LSTM_REDUCED_FEATURES) or
            rf_sequence.shape != (SEQUENCE_LENGTH, FEATURES_PER_FRAME)):
            return False, None, None
            
        return True, lstm_sequence, rf_sequence

def extract_features(frame, pose, hands):
    _, pose_results = mediapipe_detection(frame, pose)
    _, hand_results = mediapipe_detection(frame, hands)
    features = extract_arm_features(pose_results, hand_results)
    return features, pose_results, hand_results

def predict_gesture(lstm_sequence, rf_sequence):
    if len(lstm_sequence) < MIN_FRAMES_FOR_PREDICTION:
        print(f"Insufficient frames: {len(lstm_sequence)}/{MIN_FRAMES_FOR_PREDICTION}")
        return None, 0.0
    try:
        lstm_input_name = lstm_session.get_inputs()[0].name
        rf_input_name = rf_session.get_inputs()[0].name
        
        lstm_input = lstm_sequence.reshape(1, SEQUENCE_LENGTH, LSTM_REDUCED_FEATURES).astype(np.float32)
        lstm_output = lstm_session.run(None, {lstm_input_name: lstm_input})
        lstm_probs = lstm_output[0][0]
        lstm_pred = np.argmax(lstm_probs)
        lstm_conf = lstm_probs[lstm_pred]
        
        # For RF, use only the latest frame and reduce to 1000 features
        rf_latest_frame = rf_sequence[-1]  # Shape: (1260,)
        rf_reduced = rf_latest_frame[:RF_REDUCED_FEATURES]  # Placeholder reduction to 1000
        rf_input = rf_reduced.reshape(1, RF_REDUCED_FEATURES).astype(np.float32)
        rf_output = rf_session.run(None, {rf_input_name: rf_input})
        rf_pred = int(rf_output[0][0])
        
        if lstm_conf > MIN_CONFIDENCE:
            print(f"LSTM Prediction: {ACTIONS[lstm_pred]}, Confidence: {lstm_conf}")
            return ACTIONS[lstm_pred], lstm_conf
        print(f"RF Prediction: {ACTIONS[rf_pred]}, Confidence: 0.85 (fallback)")
        return ACTIONS[rf_pred], 0.85
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0.0

def generate_sentence_stepwise(start_word, model, max_length=20):
    if start_word not in all_words:
        return [start_word.capitalize() + "."]
    
    sentence = [start_word[0].upper() + start_word[1:]]
    current_ngram = (start_word,)
    
    for n in range(1, max_length):
        next_words = model[current_ngram]
        if not next_words:
            break
        possible_words = list(next_words.keys())
        weights = list(next_words.values())
        next_word = random.choices(possible_words, weights=weights, k=1)[0]
        sentence.append(next_word)
        current_ngram = tuple(sentence[-n:]) if n <= len(sentence) else tuple(sentence)
        
        if next_word in [".", "!", "?"]:
            break
    
    print(f"Generated sentence for '{start_word}': {' '.join(sentence)}")
    return sentence

def key_to_text(key_word):
    if key_word.lower() in allowed_first_words:
        return generate_sentence_stepwise(key_word.lower(), multi_ngram_model)
    print(f"Word '{key_word}' not in allowed_first_words")
    return []

def main():
    collector = RealTimeFeatureCollector()
    predictions = deque(maxlen=20)
    generated_words = []
    last_display_time = 0
    current_translation = ""
    is_recording = False
    last_pred_time = time.time()
    last_word_time = 0
    
    cv2.namedWindow("Sign Language Recognition")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        frame = cv2.flip(frame, 1)
        
        raw_features, pose_results, hand_results = extract_features(frame, pose, hands)
        
        if is_recording:
            ready, lstm_seq, rf_seq = collector.process_frame(raw_features)
            
            print(f"Buffer length: {len(collector.lstm_buffer)}")
            if (ready and time.time() - last_pred_time >= PREDICTION_WINDOW_SECONDS):
                pred, conf = predict_gesture(lstm_seq, rf_seq)
                if pred and conf >= MIN_CONFIDENCE:
                    print(f"Predicted gesture: {pred}, Confidence: {conf}")
                    if pred.lower() in allowed_first_words:
                        predictions.append(pred)
                        final_pred = max(set(predictions), key=predictions.count)
                        new_words = key_to_text(final_pred)
                        if new_words and not generated_words:
                            generated_words = [new_words[0]]
                            last_word_time = time.time()
                            print(f"Starting sentence with: {generated_words[0]}")
                        predictions.clear()
                        last_pred_time = time.time()
                    else:
                        print(f"Gesture '{pred}' ignored (not in allowed_first_words)")
        
        current_time = time.time()
        if generated_words and len(generated_words) < len(key_to_text(final_pred if predictions else generated_words[0].lower())):
            if current_time - last_word_time >= WORD_DISPLAY_INTERVAL:
                full_sentence = key_to_text(final_pred if predictions else generated_words[0].lower())
                if full_sentence and len(generated_words) < len(full_sentence):
                    generated_words.append(full_sentence[len(generated_words)])
                    last_word_time = current_time
                    print(f"Added word: {generated_words[-1]}, Current sentence: {' '.join(generated_words)}")
                    if len(generated_words) == len(full_sentence):
                        engine.say(" ".join(generated_words))
                        engine.runAndWait()
        
        if not is_recording and generated_words and (current_time - last_display_time > DISPLAY_DURATION):
            generated_words = []
        
        frame = draw_landmarks(frame, pose_results, hand_results)
        frame = draw_ui(frame, is_recording, " ".join(generated_words), current_translation, 
                       time.time() - last_pred_time if is_recording else 0)
        
        cv2.imshow("Sign Language Recognition", frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            is_recording = not is_recording
            if is_recording:
                collector = RealTimeFeatureCollector()  # Reset collector
                predictions.clear()
                generated_words = []
                last_pred_time = time.time()
                print("Recording started")
            else:
                if generated_words:
                    engine.say(f"heard {' '.join(generated_words)}")
                    engine.runAndWait()
                last_display_time = current_time
                print("Recording stopped")
        elif key == ord('v'):
            if generated_words:
                engine.say(" ".join(generated_words))
                engine.runAndWait()
        elif key == ord('c'):
            generated_words = []
    
    cap.release()
    cv2.destroyAllWindows()

def draw_landmarks(frame, pose_results, hand_results):
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=COLOR_GREEN, thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=COLOR_RED, thickness=2))
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=COLOR_GREEN, thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=COLOR_RED, thickness=2))
    return frame

def draw_ui(frame, is_recording, generated_sentence, translation, time_elapsed):
    sidebar = np.zeros((frame.shape[0], 300, 3), dtype=np.uint8)
    for i in range(300):
        alpha = i / 300
        color = tuple(int(COLOR_DARK_GRAY[j] * (1 - alpha) + COLOR_LIGHT_GRAY[j] * alpha) for j in range(3))
        sidebar[:, i] = color
    frame = np.hstack([sidebar, frame])
    
    status_color = COLOR_GREEN if is_recording else COLOR_RED
    status_text = "RECORDING" if is_recording else "STOPPED"
    cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
    cv2.line(frame, (20, 70), (280, 70), COLOR_WHITE, 1)
    
    cv2.putText(frame, f"Time: {time_elapsed:.1f}s", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)
    cv2.line(frame, (20, 120), (280, 120), COLOR_WHITE, 1)
    
    cv2.putText(frame, "Predicted Text:", (20, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
    cv2.rectangle(frame, (20, 170), (280, 250), COLOR_WHITE, 2)
    if generated_sentence:
        display_sentence = generated_sentence[-20:] if len(generated_sentence) > 20 else generated_sentence
        cv2.putText(frame, display_sentence, (30, 210), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GREEN, 2)
    else:
        cv2.putText(frame, "No Prediction", (30, 210), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 2)
    cv2.line(frame, (20, 260), (280, 260), COLOR_WHITE, 1)
    
    if translation:
        cv2.putText(frame, "Translation:", (20, 290), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
        cv2.putText(frame, translation, (20, 320), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLUE, 1)
    
    y_offset = frame.shape[0] - 120
    instructions = [
        ("[S] Start/Stop", COLOR_GREEN),
        ("[V] Voice", COLOR_BLUE),
        ("[C] Clear", COLOR_RED),
        ("[Q] Quit", COLOR_WHITE)
    ]
    for i, (text, color) in enumerate(instructions):
        cv2.putText(frame, text, (20, y_offset + i * 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame

if __name__ == "_main_":
    main()