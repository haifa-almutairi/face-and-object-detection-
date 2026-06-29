import cv2
import pandas as pd # Data handling / spreadsheet library for Python (like excel inside Python) to read and process .csv files easily
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score # Scikit-learn is a machine learning library to calculate performance metrics

# === Load labels ===
data = pd.read_csv("labels.csv")

# Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Prepare lists for true and predicted labels
y_true = []
y_pred = []

for index, row in data.iterrows():
    filename = row["filename"]
    true_label = row["has_face"]
    img_path = f"images/{filename}"

    # Read image
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"⚠️ Could not load image: {img_path}")
        continue  # Skip if image not found

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Prediction: 1 if faces detected, 0 otherwise
    pred_label = 1 if len(faces) > 0 else 0

    # Store results
    y_true.append(true_label)
    y_pred.append(pred_label)

# === Evaluation metrics ===
conf_matrix = confusion_matrix(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
accuracy = accuracy_score(y_true, y_pred)

# === Display results ===
print("\n=== Face Detection Evaluation Results ===")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")

