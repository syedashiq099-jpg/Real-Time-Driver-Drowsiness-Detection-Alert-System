from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import winsound
import os
import urllib.request
import bz2
import time


# =========================================================
# 1. Automatically download facial landmark model
# =========================================================

MODEL_FILE = "shape_predictor_68_face_landmarks.dat"
MODEL_URL = "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

# Save model in the same folder as this Python file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILE)

if not os.path.exists(MODEL_PATH):

    print("Facial landmark model not found.")
    print("Downloading model... Please wait.")

    compressed_file = os.path.join(
        BASE_DIR,
        "shape_predictor_68_face_landmarks.dat.bz2"
    )

    try:
        urllib.request.urlretrieve(MODEL_URL, compressed_file)

        print("Download completed.")
        print("Extracting model...")

        with bz2.open(compressed_file, "rb") as source:
            with open(MODEL_PATH, "wb") as target:
                target.write(source.read())

        os.remove(compressed_file)

        print("Model ready!")

    except Exception as e:
        print("Model download failed.")
        print("Error:", e)
        input("Press Enter to exit...")
        exit()


# =========================================================
# 2. Eye Aspect Ratio
# =========================================================

def eyeAspectRatio(eye):

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear


# =========================================================
# 3. Settings
# =========================================================

frequency = 2500
duration = 1000

count = 0

# Eye closure threshold
earThresh = 0.30

# Number of consecutive frames
earFrames = 48


# =========================================================
# 4. Load camera
# =========================================================

cam = cv2.VideoCapture(0)

if not cam.isOpened():

    print("Camera 0 could not be opened.")
    print("Trying camera 1...")

    cam = cv2.VideoCapture(1)

if not cam.isOpened():

    print("ERROR: Camera could not be opened.")
    input("Press Enter to exit...")
    exit()


# =========================================================
# 5. Dlib face detector and landmark predictor
# =========================================================

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(MODEL_PATH)


# =========================================================
# 6. Get eye landmark positions
# =========================================================

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]

(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


print("---------------------------------------")
print("DROWSINESS DETECTION STARTED")
print("---------------------------------------")
print("Press Q to quit.")


last_beep = 0


# =========================================================
# 7. Main loop
# =========================================================

while True:

    success, frame = cam.read()

    if not success:
        print("Could not read camera frame.")
        break

    frame = imutils.resize(frame, width=600)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)


    # -----------------------------------------------------
    # Detect face
    # -----------------------------------------------------

    for rect in rects:

        shape = predictor(gray, rect)

        shape = face_utils.shape_to_np(shape)


        # Get eyes

        leftEye = shape[lStart:lEnd]

        rightEye = shape[rStart:rEnd]


        # Calculate EAR

        leftEAR = eyeAspectRatio(leftEye)

        rightEAR = eyeAspectRatio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0


        # Draw eye contours

        leftEyeHull = cv2.convexHull(leftEye)

        rightEyeHull = cv2.convexHull(rightEye)


        cv2.drawContours(
            frame,
            [leftEyeHull],
            -1,
            (0, 0, 255),
            1
        )

        cv2.drawContours(
            frame,
            [rightEyeHull],
            -1,
            (0, 0, 255),
            1
        )


        # Display EAR value

        cv2.putText(
            frame,
            "EAR: {:.2f}".format(ear),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # -------------------------------------------------
        # Drowsiness detection
        # -------------------------------------------------

        if ear < earThresh:

            count += 1

            cv2.putText(
                frame,
                "Eyes Closed",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )


            if count >= earFrames:

                cv2.putText(
                    frame,
                    "DROWSINESS DETECTED!",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )


                # Beep only once every 2 seconds
                current_time = time.time()

                if current_time - last_beep > 2:

                    winsound.Beep(
                        frequency,
                        duration
                    )

                    last_beep = current_time


        else:

            count = 0

            cv2.putText(
                frame,
                "Awake",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


    # =====================================================
    # Show camera
    # =====================================================

    cv2.imshow("Drowsiness Detection", frame)


    # Press Q to quit

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# =========================================================
# 8. Release resources
# =========================================================

cam.release()

cv2.destroyAllWindows()

print("Drowsiness Detection stopped.")
