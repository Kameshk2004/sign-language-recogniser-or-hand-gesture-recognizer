# 🖐️ Sign Language and Hand Gesture Recognizer

A machine learning project for real-time sign language recognition and hand gesture detection using computer vision. This project aims to bridge communication gaps for the hearing impaired and enable natural human-computer interaction through intuitive hand movements.

## 📂 Key Features

* Real-time hand gesture detection using OpenCV and Mediapipe
* Dynamic and static gesture recognition with deep learning models
* Flexibility for custom gesture training
* JSON-based label management for easy customization
* High accuracy with optimized hand tracking algorithms

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* Required libraries: `opencv-python`, `mediapipe`, `numpy`, `tensorflow`, `scikit-learn`

### Installation

Clone this repository and install the required packages:

```bash
git clone https://github.com/your-username/sign-language-recognizer.git
cd sign-language-recognizer
pip install -r requirements.txt
```

### Project Structure

```
.
├── collect_data.py             # Script to collect gesture data for training
├── labels.json                 # JSON file for mapping gesture labels
├── main.py                     # Main script to run the application
├── Project Overview.pdf        # Project documentation and overview
├── recognize_gesture.py        # Script for real-time gesture recognition
├── setup.py                    # Initial setup and environment configuration
├── train_model.py              # Model training script
├── requirements.txt            # List of required Python packages
├── README.md                   # Project documentation
└── LICENSE                     # Project license
```

### Collecting Gesture Data

1. Run the data collection script:

```bash
python collect_data.py
```

2. Follow the on-screen instructions to capture various gestures.
3. Update **labels.json** to match the collected gestures.

### Training the Model

Train the model using the collected gesture data:

```bash
python train_model.py
```

### Recognizing Gestures in Real-Time

Run the real-time gesture recognition script:

```bash
python recognize_gesture.py
```

### 📊 Model Evaluation

Includes accuracy, precision, recall, and confusion matrix for model evaluation.

## 🔗 Future Work

* Support for more sign languages
* Integration with speech synthesis for seamless communication
* Real-time performance optimization

## 🤝 Contributing

Feel free to open issues, share ideas, or submit pull requests to improve this project!

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## ⭐ Show Your Support

If you found this project helpful, please give it a star ⭐ and share it with others!

## 📧 Contact

For any questions or collaboration, reach out to me via email at kameshk0011@gmail.com.
