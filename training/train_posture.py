import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
DATA_FILE = "dataset/user_posture_data.csv"
MODEL_SAVE_PATH = "models/posture_classifier.pkl"

def train_model():
    if not os.path.exists(DATA_FILE):
        print(f"No data found at {DATA_FILE}. Please run dataset_generator.py first.")
        return

    # Load data
    df = pd.read_csv(DATA_FILE)
    X = df.iloc[:, :-1].values # All columns except the last (keypoints)
    y = df.iloc[:, -1].values  # Last column (label)

    # Split: 70% train, 20% validation, 10% test
    # First split 10% for test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    # Then split 80/20 of the rest for train/val (resulting in roughly 72/18 total)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.22, random_state=42)

    print(f"Training on {len(X_train)} samples, validating on {len(X_val)}, testing on {len(X_test)}")

    # Define model: Multi-layer Perceptron
    model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, activation='relu', solver='adam', random_state=42)

    # Train
    model.fit(X_train, y_train)

    # Evaluate on Validation
    val_score = model.score(X_val, y_val)
    print(f"Validation Accuracy: {val_score:.4f}")

    # Evaluate on Test
    y_pred = model.predict(X_test)
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('training/confusion_matrix.png')
    print("Confusion matrix saved to training/confusion_matrix.png")

    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
