import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# ✅ Load your trained Mini-XCEPTION model
model_path = r"face-Emotion--Ai\Prajwal bawane\fer2013_mini_XCEPTION.119-0.65.hdf5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = load_model(model_path, compile=False)

# ✅ Emotion labels (should match model output order)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ✅ Load OpenCV Haarcascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ✅ Open webcam
cap = cv2.VideoCapture(0)
print("[INFO] Press 'q' to quit...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip frame for mirror effect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ✅ Detect multiple faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # ✅ Extract face region
        face_roi = gray[y:y+h, x:x+w]

        try:
            # Preprocess the face for emotion model
            face_resized = cv2.resize(face_roi, (48, 48))
            face_normalized = face_resized / 255.0
            face_input = np.expand_dims(face_normalized, axis=[0, -1])  # (1, 48, 48, 1)

            # Predict emotion
            predictions = model.predict(face_input, verbose=0)
            emotion = emotion_labels[np.argmax(predictions)]

            # ✅ Draw bounding box and emotion label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        except Exception as e:
            print("Face error:", e)
            continue

    # ✅ Show output
    cv2.imshow("Multi-Face Emotion Detection (Haarcascade)", frame)

    # ✅ Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Clean up
cap.release()
cv2.destroyAllWindows()
