import os
import cv2
import numpy as np
import mediapipe as mp

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CLIPS_DIR = os.path.join(PROJECT_ROOT, 'data', 'MSASL', 'MSASL_clips')
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'MSASL', 'MSASL_keypoints')

mp_holistic = mp.solutions.holistic

os.makedirs(OUT_DIR, exist_ok=True)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

for label in os.listdir(CLIPS_DIR):
    label_dir = os.path.join(CLIPS_DIR, label)
    out_label_dir = os.path.join(OUT_DIR, label)
    os.makedirs(out_label_dir, exist_ok=True)

    for vid_name in os.listdir(label_dir):
        vid_path = os.path.join(label_dir, vid_name)

        cap = cv2.VideoCapture(vid_path)
        seq = []

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                keypoints = extract_keypoints(results)
                seq.append(keypoints)

        cap.release()

        if len(seq) == 0:
            continue

        seq = np.array(seq)

        out_path = os.path.join(out_label_dir, vid_name.replace('.mp4', '.npy'))
        np.save(out_path, seq)

print("Done extracting keypoints")