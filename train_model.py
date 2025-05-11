import os
import numpy as np
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def train_model():
    data, labels = [], []

    for file in os.listdir("dataset"):
        if file.endswith(".npy"):
            label = file.split(".")[0]
            temp_data = np.load(f"dataset/{file}")
            data.append(temp_data)
            labels.extend([label] * temp_data.shape[0])

    data = np.concatenate(data)
    labels = np.array(labels)

    with open("labels.json", "r") as f:
        labels_dict = json.load(f)

    label_to_int = {v: k for k, v in labels_dict.items()}
    int_labels = np.array([label_to_int[label] for label in labels])

    y_categorical = to_categorical(int_labels, num_classes=len(labels_dict))

    x_train, x_test, y_train, y_test = train_test_split(data, y_categorical, test_size=0.2, shuffle=True, stratify=int_labels)

    model = Sequential([
        Dense(256, activation='relu', input_shape=(x_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(labels_dict), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_test, y_test))
    model.save("models/gesture_model.keras")
    print("Model trained and saved successfully.")
