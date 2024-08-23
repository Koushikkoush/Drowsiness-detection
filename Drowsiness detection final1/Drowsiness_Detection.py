import cv2
# cv2: This imports the OpenCV library, which is used for computer vision tasks.
import mediapipe as mp
# mediapipe as mp: This imports the MediaPipe library, which is used for various machine learning tasks, including face and hand tracking.
import numpy as np
# numpy as np: This imports the NumPy library, which is used for numerical computations.
from scipy.spatial import distance
# distance: This imports the distance module from SciPy, which is used for calculating distances between points.
from pygame import mixer
# mixer: This imports the mixer module from Pygame, which is used for sound playback.
import imutils
# In summary, imutils is used to make the image processing code simpler and more readable. 


# Initialize pygame mixer
mixer.init()
# Load the music file for playback
mixer.music.load("music.wav")

# Define a function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Set thresholds and frame check parameters
thresh = 0.25
frame_check = 30

# Initialize mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize the video capture
cap = cv2.VideoCapture(0)
flag = 0

# cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Frame", 800, 600)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with mediapipe
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Get coordinates of left and right eyes
            left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
            right_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]

            # Convert to numpy arrays
            left_eye = np.array([(p.x * frame.shape[1], p.y * frame.shape[0]) for p in left_eye], dtype=np.int32)
            right_eye = np.array([(p.x * frame.shape[1], p.y * frame.shape[0]) for p in right_eye], dtype=np.int32)

            # Calculate EAR for both eyes
            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw contours around the eyes
            cv2.drawContours(frame, [left_eye], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye], -1, (0, 255, 0), 1)

            # Check if EAR is below the threshold
            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "****************ALERT!****************", (10, 325),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    mixer.music.play()
            else:
                flag = 0

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()