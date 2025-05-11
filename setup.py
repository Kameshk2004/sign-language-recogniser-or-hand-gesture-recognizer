import os
import json

def setup_project_structure():
    # Define directory structure
    directories = [
        "dataset",  # Directory for gesture data
        "models"    # Directory for saved models
    ]

    # Create directories if they don't exist
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory '{directory}' created or already exists.")

    # Create labels.json if it doesn't exist
    labels_file = "labels.json"
    if not os.path.exists(labels_file):
        with open(labels_file, "w") as f:
            json.dump({}, f)
        print(f"File '{labels_file}' created.")
    else:
        print(f"File '{labels_file}' already exists.")

if __name__ == "__main__":
    setup_project_structure()
