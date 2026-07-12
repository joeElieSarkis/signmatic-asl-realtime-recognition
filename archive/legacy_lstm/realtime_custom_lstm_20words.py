import os
import cv2
import numpy as np
import mediapipe as mp
import subprocess
from collections import deque
from tensorflow.keras.models import load_model

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_custom_model_20words_idle.h5')

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
    'Please',
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
    'Idle'
]

SEQUENCE_LENGTH = 30
CONF_THRESHOLD = 0.88
STABLE_FRAMES = 5
COOLDOWN_FRAMES = 12
DISPLAY_HOLD_FRAMES = 30
MAX_SENTENCE_WORDS = 12

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

last_spoken_text = None

def speak_text_windows(text):
    if not text:
        return

    safe_text = text.replace("'", "''")
    ps_command = (
        "Add-Type -AssemblyName System.Speech;"
        "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$speak.Speak('{safe_text}')"
    )

    subprocess.Popen(
        ["powershell", "-Command", ps_command],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

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

def hands_present(results):
    return (results.left_hand_landmarks is not None) or (results.right_hand_landmarks is not None)

def prob_viz(res, labels, frame):
    output = frame.copy()
    colors = [
        (245,117,16),
        (117,245,16),
        (16,117,245),
        (255,0,0),
        (0,255,255),
        (255,0,255),
        (128,128,255),
        (255,128,0)
    ]

    top_indices = np.argsort(res)[::-1][:8]

    for row, i in enumerate(top_indices):
        prob = float(res[i])
        color = colors[row % len(colors)]
        y1 = 110 + row * 28
        y2 = y1 + 20
        cv2.rectangle(output, (0, y1), (int(prob * 260), y2), color, -1)
        cv2.putText(output, f"{labels[i]}: {prob:.2f}", (5, y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

    return output

def context_word(raw_word, sentence):
    prev_word = sentence[-1] if sentence else None

    if raw_word == 'We' and prev_word == 'Give':
        return 'Us'

    if raw_word == 'Work' and prev_word == 'Have':
        return 'Worked'

    return raw_word

def main():
    global last_spoken_text

    model = load_model(MODEL_PATH)

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    pred_history = deque(maxlen=STABLE_FRAMES)

    accepted_text = "Waiting..."
    sentence = []
    cooldown = 0
    display_hold = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            kp = extract_keypoints(results)
            sequence.append(kp)

            current_probs = np.zeros(len(CLASSES), dtype=np.float32)

            if cooldown > 0:
                cooldown -= 1

            if display_hold > 0:
                display_hold -= 1
            elif not hands_present(results):
                accepted_text = "Waiting..."
                last_spoken_text = None

            if len(sequence) == SEQUENCE_LENGTH:
                current_probs = model.predict(np.expand_dims(np.array(sequence), axis=0), verbose=0)[0]
                pred_idx = int(np.argmax(current_probs))
                pred_label = CLASSES[pred_idx]
                pred_conf = float(current_probs[pred_idx])

                pred_history.append(pred_idx)

                stable = len(pred_history) == STABLE_FRAMES and len(set(pred_history)) == 1
                allow_prediction = hands_present(results)

                if stable and cooldown == 0 and allow_prediction and pred_conf >= CONF_THRESHOLD:
                    if pred_label != 'Idle':
                        final_word = context_word(pred_label, sentence)
                        accepted_text = final_word
                        sentence.append(final_word)

                        if len(sentence) > MAX_SENTENCE_WORDS:
                            sentence = sentence[-MAX_SENTENCE_WORDS:]

                        display_hold = DISPLAY_HOLD_FRAMES
                        cooldown = COOLDOWN_FRAMES
                        sequence.clear()
                        pred_history.clear()

                        if final_word != last_spoken_text:
                            speak_text_windows(final_word)
                            last_spoken_text = final_word

                if not allow_prediction:
                    pred_history.clear()

            image = prob_viz(current_probs, CLASSES, image)

            cv2.rectangle(image, (0, 0), (1400, 50), (50, 50, 50), -1)
            cv2.putText(image, f"Output: {accepted_text}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            cv2.rectangle(image, (0, 50), (1400, 95), (30, 30, 30), -1)
            cv2.putText(image, "Sentence: " + " ".join(sentence), (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow("Custom ASL Realtime", image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                sentence = []
                accepted_text = "Waiting..."
                last_spoken_text = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()