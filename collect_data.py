import os
import cv2
import mediapipe as mp
import numpy as np
import json

def collect_gesture_data():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.75)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    def normalize_landmarks(landmarks):
        x_coords = [lm[0] for lm in landmarks]
        y_coords = [lm[1] for lm in landmarks]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        return [
            ((x - min_x) / (max_x - min_x + 1e-5), (y - min_y) / (max_y - min_y + 1e-5))
            for x, y in landmarks
        ]

    with open("labels.json", "r") as f:
        labels_dict = json.load(f)

    label = input("Enter the label for the gesture (e.g., A, Hello): ")
    if label not in labels_dict.values():
        labels_dict[len(labels_dict)] = label
        with open("labels.json", "w") as f:
            json.dump(labels_dict, f)

    print(f"Collecting data for gesture: {label}. Press 'q' to stop.")
    data = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                    normalized = normalize_landmarks(landmarks)
                    data.append(np.array(normalized).flatten())

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    file_path = f"dataset/{label}.npy"
    if os.path.exists(file_path):
        existing_data = np.load(file_path)
        if existing_data.size > 0:
            data = np.concatenate([existing_data, data], axis=0)

    np.save(file_path, np.array(data))
    print(f"Data for gesture '{label}' saved successfully.")
