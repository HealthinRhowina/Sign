import os
import numpy as np
from keras.models import Sequential
from keras.layers import (LSTM, Dense, Dropout, Bidirectional, 
                         Conv1D, BatchNormalization, MaxPooling1D)
from keras.optimizers import Adam
from keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                            ModelCheckpoint, TensorBoard)
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, 
                            classification_report, confusion_matrix)
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
import tf2onnx
import onnx

# Configuration aligned with process_data.py
DATA_PATH = "zdata1"  # Updated to match your directory
SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 5040  # Updated to match process_data.py output
REDUCED_FEATURES = 800  # Increased to retain more variance
BATCH_SIZE = 32
EPOCHS = 200

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
def load_data():
    X, y = [], []
    
    for action_idx, action in enumerate(ACTIONS):
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            print(f"Skipping missing action: {action}")
            continue
            
        sequence_folders = sorted([f for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f))])
        
        for seq_folder in sequence_folders:
            seq_path = os.path.join(action_path, seq_folder)
            try:
                sequence = np.stack([np.load(os.path.join(seq_path, f'{i}.npy')) 
                                   for i in range(SEQUENCE_LENGTH)]).astype(np.float32)
                if sequence.shape != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
                    print(f"Skipping invalid shape {sequence.shape} in {seq_path}")
                    continue
                X.append(sequence)
                y.append(action_idx)
            except Exception as e:
                print(f"Error loading {seq_path}: {e}")
    
    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=len(ACTIONS))
    return X, y

def reduce_dimensionality(X, n_components=REDUCED_FEATURES):
    num_samples, seq_length, num_features = X.shape
    X_flat = X.reshape(num_samples * seq_length, num_features)
    pca = PCA(n_components=n_components, whiten=True)
    X_reduced = pca.fit_transform(X_flat)
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    return X_reduced.reshape(num_samples, seq_length, n_components)

def create_lstm_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(512, 5, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling1D(3),
        Dropout(0.3),
        
        Conv1D(256, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        
        Bidirectional(LSTM(256, return_sequences=True)),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(LSTM(64)),
        
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=3e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'top_k_categorical_accuracy'])
    return model

def train_lstm_model(X, y):
    y_indices = np.argmax(y, axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_indices, test_size=0.2, stratify=y_indices, random_state=42
    )
    
    print("\nReducing dimensionality using PCA...")
    X_train = reduce_dimensionality(X_train)
    X_test = reduce_dimensionality(X_test)
    
    print("\nApplying Random Oversampling...")
    ros = RandomOverSampler(random_state=42)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_train_bal, y_train_bal = ros.fit_resample(X_train_flat, y_train)
    X_train_bal = X_train_bal.reshape(-1, SEQUENCE_LENGTH, REDUCED_FEATURES)
    
    y_train_bal_cat = to_categorical(y_train_bal, num_classes=len(ACTIONS))
    y_test_cat = to_categorical(y_test, num_classes=len(ACTIONS))
    
    lstm_model = create_lstm_model((SEQUENCE_LENGTH, REDUCED_FEATURES), len(ACTIONS))
    
    callbacks = [
        EarlyStopping(patience=30, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=10, min_lr=1e-6),
        ModelCheckpoint('best_lstm_pca.h5', save_best_only=True, monitor='val_accuracy'),
        TensorBoard(log_dir='./logs')
    ]
    
    print("\nTraining LSTM model...")
    lstm_model.fit(
        X_train_bal, y_train_bal_cat,
        validation_data=(X_test, y_test_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    return lstm_model, X_test, y_test_cat

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print(f"\n{'='*40}")
    print("LSTM Model Evaluation")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2%}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=ACTIONS))
    
    cm = confusion_matrix(y_true, y_pred)
    np.save("confusion_matrix_lstm_pca.npy", cm)

if __name__ == "__main__":
    X, y = load_data()
    
    if X.shape[0] == 0:
        raise ValueError("No valid training data found! Check zdata1 directory.")
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.sum(y, axis=0)}")
    
    try:
        lstm_model, X_test, y_test = train_lstm_model(X, y)
    except MemoryError:
        print("\nMemory Error! Reduce PCA components or data size.")
        exit(1)
    except ValueError as e:
        print(f"\nError during training: {e}")
        exit(1)
    
    evaluate_model(lstm_model, X_test, y_test)
    lstm_model.save("sign_language_lstm_pca.h5")
    
    input_signature = [tf.TensorSpec([None, SEQUENCE_LENGTH, REDUCED_FEATURES], tf.float32, name='input')]
    onnx_model, _ = tf2onnx.convert.from_keras(lstm_model, input_signature=input_signature, opset=13)
    with open("cnn_lstm_pca.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    onnx.checker.check_model(onnx.load("cnn_lstm_pca.onnx"))
    print("ONNX model validated successfully!")