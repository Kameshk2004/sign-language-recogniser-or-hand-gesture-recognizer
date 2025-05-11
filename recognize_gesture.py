import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json

def run_gesture_recognition():
    model = tf.keras.models.load_model("models/gesture_model.keras")

    with open("labels.json", "r") as f:
        labels_dict = json.load(f)

    int_to_label = {int(k): v for k, v in labels_dict.items()}

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, 
                           min_detection_confidence=0.75, min_tracking_confidence=0.75)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), 'Failed to open the camera.'

    def normalize_landmarks(landmarks):
        x_coords = [lm[0] for lm in landmarks]
        y_coords = [lm[1] for lm in landmarks]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        return [
            ((x - min_x) / (max_x - min_x + 1e-5), (y - min_y) / (max_y - min_y + 1e-5))
            for x, y in landmarks
        ]

    def get_prediction(normalized_landmarks):
        prediction = model.predict(np.array([np.array(normalized_landmarks).flatten()]))
        confidence_threshold = 0.85
        max_confidence = np.max(prediction)
        if max_confidence < confidence_threshold:
            return "", max_confidence
        label_index = np.argmax(prediction[0])
        return int_to_label.get(label_index, ""), max_confidence

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                normalized = normalize_landmarks(landmarks)
                label, confidence = get_prediction(normalized)

                if label:
                    cv2.putText(frame, f"{label} ({confidence:.2f})", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
