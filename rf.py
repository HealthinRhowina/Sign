import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import shutil

# Configuration aligned with process_data.py
DATA_PATH = "zdata1"  # Updated to match your directory
SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 5040  # Updated to match process_data.py output
TEMP_DIR = "temp_rf_data"
ONNX_MODEL_PATH = "new_rf.onnx"

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

def save_rf_data_to_disk():
    os.makedirs(TEMP_DIR, exist_ok=True)
    total_samples = 0
    
    for action_idx, action in enumerate(ACTIONS):
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            print(f"Skipping missing folder: {action_path}")
            continue

        sequence_folders = sorted(os.listdir(action_path))
        
        for seq_idx, seq_folder in enumerate(sequence_folders):
            seq_path = os.path.join(action_path, seq_folder)
            try:
                sequence = np.stack([np.load(os.path.join(seq_path, f"{i}.npy")) 
                                   for i in range(SEQUENCE_LENGTH)])
                if sequence.shape != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
                    print(f"Skipping invalid shape {sequence.shape} in {seq_path}")
                    continue
                
                X_2d_sample = sequence.reshape(-1)
                np.save(os.path.join(TEMP_DIR, f"sample_{total_samples}_X.npy"), X_2d_sample)
                np.save(os.path.join(TEMP_DIR, f"sample_{total_samples}_y.npy"), np.array([action_idx]))
                total_samples += 1
            except FileNotFoundError as e:
                print(f"Warning: Skipping {seq_path} due to missing file: {e}")
                continue
    
    print(f"Total samples saved: {total_samples}")
    return total_samples

def load_rf_data(indices):
    X_2d_list, y_list = [], []
    for idx in indices:
        x_path = os.path.join(TEMP_DIR, f"sample_{idx}_X.npy")
        y_path = os.path.join(TEMP_DIR, f"sample_{idx}_y.npy")
        if os.path.exists(x_path) and os.path.exists(y_path):
            X_2d_list.append(np.load(x_path))
            y_list.append(np.load(y_path)[0])
        else:
            print(f"Missing file for sample {idx}, skipping.")

    if len(X_2d_list) == 0:
        print("Error: No valid training data found.")
        return np.array([]), np.array([])

    return np.array(X_2d_list, dtype=np.float32), np.array(y_list)

def train_rf_and_convert_onnx():
    if not os.path.exists(TEMP_DIR) or len(os.listdir(TEMP_DIR)) == 0:
        print("Error: No data found. Run `save_rf_data_to_disk()` first.")
        return

    indices = np.arange(len(os.listdir(TEMP_DIR)) // 2)
    y = np.array([np.load(os.path.join(TEMP_DIR, f"sample_{idx}_y.npy"))[0] for idx in indices])

    if len(indices) == 0:
        print("Error: No training samples found.")
        return

    train_indices, test_indices = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    X_train_2d, y_train = load_rf_data(train_indices)
    X_test_2d, y_test = load_rf_data(test_indices)

    if X_train_2d.size == 0 or y_train.size == 0:
        print("Error: No training data loaded.")
        return

    print(f"Training data: {X_train_2d.shape[0]} samples with {X_train_2d.shape[1]} features each.")

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=1000, random_state=42)
    X_train_2d_pca = pca.fit_transform(X_train_2d)
    X_test_2d_pca = pca.transform(X_test_2d)
    print(f"Reduced to {X_train_2d_pca.shape[1]} features with PCA.")

    unique, counts = np.unique(y_train, return_counts=True)
    min_class_size = min(counts)

    if min_class_size < 6:
        print(f"Warning: Some classes have fewer than 6 samples. SMOTE skipped.")
        X_train_2d_bal, y_train_bal = X_train_2d_pca, y_train
    else:
        smote = SMOTE(k_neighbors=5, random_state=42)
        X_train_2d_bal, y_train_bal = smote.fit_resample(X_train_2d_pca, y_train)

    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,  # Reduced from 1000 to avoid MemoryError
        max_depth=20,      # Reduced from 50 to make trees smaller
        min_samples_split=5,  # Increased from 2 to prune trees
        max_features='sqrt',
        class_weight='balanced_subsample',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    rf_model.fit(X_train_2d_bal, y_train_bal)

    rf_preds = rf_model.predict(X_test_2d_pca)
    print("\nRandom Forest Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, rf_preds):.2%}")
    print(f"Macro F1: {f1_score(y_test, rf_preds, average='macro'):.2%}")
    print(classification_report(y_test, rf_preds, target_names=ACTIONS, zero_division=0))
    np.save("confusion_matrix_rf.npy", confusion_matrix(y_test, rf_preds))

    print("\nConverting Random Forest to ONNX format...")
    initial_type = [("float_input", FloatTensorType([None, 1000]))]  # Updated for PCA output
    onnx_model = convert_sklearn(rf_model, initial_types=initial_type)

    with open(ONNX_MODEL_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Model successfully converted and saved as {ONNX_MODEL_PATH}")

def main():
    print("\nSaving dataset...")
    total_samples = save_rf_data_to_disk()

    if total_samples == 0:
        print("Error: No valid samples found. Exiting.")
        return

    print("\nTraining model and converting to ONNX...")
    train_rf_and_convert_onnx()

    shutil.rmtree(TEMP_DIR)
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()
    