# Real-Time-Driver-Drowsiness-Detection-Alert-System

A real-time computer vision project that detects driver drowsiness by monitoring eye closure using Eye Aspect Ratio (EAR). The system uses Python, OpenCV, and dlib facial landmarks to track the eyes and triggers an audible buzzer alert when prolonged eye closure is detected.

## 🛠️ Technologies Used

- **Python** – Core programming language
- **OpenCV** – Real-time video processing and image handling
- **dlib** – Face detection and facial landmark detection
- **imutils** – Image processing and frame resizing utilities
- **NumPy** – Numerical and array operations
- **SciPy** – Euclidean distance calculation for Eye Aspect Ratio (EAR)
- **WinSound** – Audible buzzer alert when drowsiness is detected
- **Webcam** – Real-time video input

## ✨ Key Features

- 🎥 **Real-Time Webcam Monitoring** – Continuously monitors the user's face through a webcam.
- 👤 **Face Detection** – Detects faces in real-time using dlib's frontal face detector.
- 👁️ **Facial Landmark Detection** – Identifies 68 facial landmark points to locate the eyes.
- 📊 **Eye Aspect Ratio (EAR)** – Calculates the EAR to monitor eye opening and closing.
- 😴 **Drowsiness Detection** – Detects prolonged eye closure based on a predefined EAR threshold.
- 🔊 **Automatic Buzzer Alert** – Activates an audible alarm when drowsiness is detected.
- 👀 **Real-Time Eye Tracking** – Displays eye contours around both eyes during monitoring.
- 📈 **Live EAR Display** – Shows the calculated EAR value on the video screen.
- ⚡ **Real-Time Processing** – Performs face and eye analysis continuously with webcam frames.
- ❌ **Easy Exit** – Press `Q` to safely stop the application.

- ## 🎯 How It Works

1. The webcam captures live video frames.
2. The system detects the user's face using dlib.
3. Facial landmarks are detected around the eyes.
4. The Eye Aspect Ratio (EAR) is calculated for both eyes.
5. If the EAR remains below the threshold for a specified number of consecutive frames, the system identifies possible drowsiness.
6. An audible buzzer alert is triggered to warn the user.

## 💻 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/syedashiq099-jpg/Real-Time-Driver-Drowsiness-Detection-Alert-System.git
