import os
import sys
from collect_data import collect_gesture_data
from train_model import train_model
from recognize_gesture import run_gesture_recognition
import json

def main():
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    if not os.path.exists("labels.json"):
        with open("labels.json", "w") as f:
            json.dump({}, f)

    print("Select an option:")
    print("1. Collect gesture data")
    print("2. Train model")
    print("3. Run gesture recognition")
    choice = input("Enter your choice (1/2/3): ")

    if choice == "1":
        collect_gesture_data()
    elif choice == "2":
        train_model()
    elif choice == "3":
        run_gesture_recognition()
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    main()
