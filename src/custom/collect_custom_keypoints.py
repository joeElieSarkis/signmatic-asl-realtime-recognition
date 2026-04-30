import os
import cv2
import numpy as np
import mediapipe as mp

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'Custom', 'custom_keypoints')

CLASSES = [
    'Nice',
    'Eat',
    'Yes',
    'No',
    'Water',
    'Help',
    'Hello',
    'Fine',
    'Good',
    'Please', ###  tfakera like 
    'Give',
    'We',
    'A',
    'Have',
    'Work',
    'So',
    'Hard',
    'Live',
    'Love',
    'Thanks',
    'High',
    'Grade',
    'Lebanese',
    'International',
    'University',
    'Teacher',
    'Happy',   ### am yfakera nice
    'Like', 
    'Want',
    'Deaf',
    'School',
    'What',
    'Need',
    'Friend',
    'Learn',
    'Book',
    'Computer',
    'Idle'
]

SEQUENCES_PER_CLASS = {
    'Nice': 330,  
    'Eat': 250,  
    'Yes': 330,  
    'No': 350, 
    'Water': 350,   
    'Help': 250,  
    'Hello': 270,    
    'Fine': 290,     
    'Good': 290,    
    'Please': 310,   
    'Give': 290,   
    'We': 340,
    'A': 340,
    'Have': 250,
    'Work': 250,
    'So': 250,
    'Hard': 250,
    'Live': 250,
    'Love': 250,
    'Thanks': 250,
    'High': 290,
    'Grade': 290,
    'Lebanese': 290,
    'International': 290,
    'University': 270,
    'Teacher': 220,
    'Happy': 260,
    'Like': 220,
    'Want': 220,
    'Deaf': 220,
    'School': 220,
    'What': 220,
    'Need': 220,
    'Friend': 220,
    'Learn': 220,
    'Book': 220,
    'Computer': 220,
    'Idle': 300
}

SEQUENCE_LENGTH = 30

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(frame, model):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh]).astype(np.float32)

def get_next_sequence_index(class_dir):
    if not os.path.exists(class_dir):
        return 0

    existing = [
        d for d in os.listdir(class_dir)
        if os.path.isdir(os.path.join(class_dir, d)) and d.isdigit()
    ]

    if not existing:
        return 0

    return max(int(x) for x in existing) + 1

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)

    for cls in CLASSES:
        os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

def main():
    ensure_dirs()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for cls in CLASSES:
            class_dir = os.path.join(DATA_DIR, cls)
            start_seq = get_next_sequence_index(class_dir)
            target_total = SEQUENCES_PER_CLASS[cls]

            print(f"\nCollecting class: {cls}")
            print(f"Starting from sequence {start_seq}, target total = {target_total}")

            for seq_idx in range(start_seq, target_total):
                seq_dir = os.path.join(class_dir, str(seq_idx))
                os.makedirs(seq_dir, exist_ok=True)

                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()

                    if not ret:
                        print("Camera read failed.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    if frame_num == 0:
                        cv2.putText(image, f'CLASS: {cls}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(image, f'SEQUENCE: {seq_idx + 1}/{target_total}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(image, 'GET READY', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow('Custom Data Collection', image)

                        key = cv2.waitKey(1200)

                        if key & 0xFF == ord('q'):
                            cap.release()
                            cv2.destroyAllWindows()
                            return

                    ret, frame = cap.read()

                    if not ret:
                        print("Camera read failed.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    keypoints = extract_keypoints(results)
                    np.save(os.path.join(seq_dir, f'{frame_num}.npy'), keypoints)

                    cv2.putText(image, f'CLASS: {cls}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f'SEQUENCE: {seq_idx + 1}/{target_total}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f'FRAME: {frame_num + 1}/{SEQUENCE_LENGTH}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('Custom Data Collection', image)

                    key = cv2.waitKey(1)

                    if key & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

    cap.release()
    cv2.destroyAllWindows()
    print("Done collecting custom data.")

if __name__ == '__main__':
    main()