import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import subprocess
from collections import deque
from tensorflow.keras.models import load_model

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'final_signmatic_transformer_50words.h5')

CLASSES = [
    'Nice', 'Eat', 'Yes', 'No', 'Water', 'Help', 'Hello', 'Fine', 'Good', 'Please',
    'Give', 'We', 'A', 'Have', 'Work', 'So', 'Hard', 'Live', 'Love', 'Thanks',
    'High', 'Grade', 'Lebanese', 'International', 'University',
    'Teacher', 'Happy', 'Like', 'Want', 'Deaf', 'School',
    'What', 'Need', 'Friend', 'Learn', 'Book', 'Computer',
    'Again', 'Father', 'Mother', 'Where', 'Forget', 'Nothing',
    'I', 'You', 'And', 'My', 'Name', 'Is', 'ILoveYou',
    'Idle'
]

SEQUENCE_LENGTH = 30
CONF_THRESHOLD = 0.88
STABLE_FRAMES = 5
COOLDOWN_FRAMES = 6
DISPLAY_HOLD_FRAMES = 18
RECENT_HANDS_FRAMES = 8
DISPLAY_SENTENCE_WORDS = 10

WINDOW_W = 800
WINDOW_H = 600

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

last_spoken_text = None

PAST_AFTER_HAVE = {
    'Work': 'worked',
    'Want': 'wanted',
    'Need': 'needed',
    'Learn': 'learned',
    'Love': 'loved',
    'Like': 'liked',
    'Help': 'helped',
    'Eat': 'eaten',
    'Give': 'given',
    'Forget': 'forgotten'
}

OBJECT_US_PREVIOUS = {
    'Give',
    'Help'
}

DISPLAY_WORDS = {
    'Nice': 'nice',
    'Eat': 'eat',
    'Yes': 'yes',
    'No': 'no',
    'Water': 'water',
    'Help': 'help',
    'Hello': 'hello',
    'Fine': 'fine',
    'Good': 'good',
    'Please': 'please',
    'Give': 'give',
    'We': 'we',
    'A': 'A',
    'Have': 'have',
    'Work': 'work',
    'So': 'so',
    'Hard': 'hard',
    'Live': 'live',
    'Love': 'love',
    'Thanks': 'thanks',
    'High': 'high',
    'Grade': 'grade',
    'Lebanese': 'Lebanese',
    'International': 'International',
    'University': 'University',
    'Teacher': 'teacher',
    'Happy': 'happy',
    'Like': 'like',
    'Want': 'want',
    'Deaf': 'deaf',
    'School': 'school',
    'What': 'what',
    'Need': 'need',
    'Friend': 'friend',
    'Learn': 'learn',
    'Book': 'book',
    'Computer': 'computer',
    'Again': 'again',
    'Father': 'father',
    'Mother': 'mother',
    'Where': 'where',
    'Forget': 'forget',
    'Nothing': 'nothing',
    'I': 'I',
    'You': 'you',
    'And': 'and',
    'My': 'my',
    'Name': 'name',
    'Is': 'is',
    'ILoveYou': 'I love you',
    'Idle': ''
}

def speech_word(display_text):
    if display_text == 'live':
        return 'liv'

    return display_text

def speak_text(text):
    if not text:
        return

    spoken_text = speech_word(text)
    safe_text = spoken_text.replace("'", "''")

    if sys.platform.startswith('win'):
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
    else:
        subprocess.Popen(
            ["espeak-ng", safe_text],
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

def apply_context(raw_word, raw_sentence):
    previous_raw = raw_sentence[-1] if raw_sentence else None

    if raw_word == 'We' and previous_raw in OBJECT_US_PREVIOUS:
        return 'us'

    if previous_raw == 'Have' and raw_word in PAST_AFTER_HAVE:
        return PAST_AFTER_HAVE[raw_word]

    return DISPLAY_WORDS.get(raw_word, raw_word.lower())

def clean_sentence(words):
    if not words:
        return ''

    text = ' '.join(words)
    text = text.replace('I love you', 'I love you')

    if len(text) > 0:
        text = text[0].upper() + text[1:]

    return text

def sentence_line(sentence_words):
    visible = sentence_words[-DISPLAY_SENTENCE_WORDS:]
    return "Sentence: " + clean_sentence(visible)

def make_display(frame, accepted_text, sentence_words):
    frame = cv2.resize(frame, (WINDOW_W, WINDOW_H))

    cv2.rectangle(frame, (0, 0), (WINDOW_W, 60), (50, 50, 50), -1)
    cv2.putText(
        frame,
        f"Output: {accepted_text}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255,255,255),
        2,
        cv2.LINE_AA
    )

    cv2.rectangle(frame, (0, WINDOW_H - 65), (WINDOW_W, WINDOW_H), (30, 30, 30), -1)
    cv2.putText(
        frame,
        sentence_line(sentence_words),
        (20, WINDOW_H - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2,
        cv2.LINE_AA
    )

    return frame

def main():
    global last_spoken_text

    model = load_model(MODEL_PATH)

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    pred_history = deque(maxlen=STABLE_FRAMES)
    recent_hands = deque(maxlen=RECENT_HANDS_FRAMES)

    accepted_text = "Waiting..."
    sentence_words = []
    raw_sentence = []

    cooldown = 0
    display_hold = 0

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    cv2.namedWindow("Hybrid ASL Realtime", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hybrid ASL Realtime", WINDOW_W, WINDOW_H)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            current_hands = hands_present(results)
            recent_hands.append(current_hands)

            kp = extract_keypoints(results)
            sequence.append(kp)

            if cooldown > 0:
                cooldown -= 1

            if display_hold > 0:
                display_hold -= 1
            elif not any(recent_hands):
                accepted_text = "Waiting..."
                last_spoken_text = None

            if len(sequence) == SEQUENCE_LENGTH:
                probs = model.predict(np.expand_dims(np.array(sequence), axis=0), verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                pred_label = CLASSES[pred_idx]
                pred_conf = float(probs[pred_idx])

                pred_history.append(pred_idx)

                stable = len(pred_history) == STABLE_FRAMES and len(set(pred_history)) == 1
                had_recent_hands = any(recent_hands)

                if stable and cooldown == 0 and had_recent_hands and pred_conf >= CONF_THRESHOLD:
                    if pred_label != 'Idle':
                        display_word = apply_context(pred_label, raw_sentence)

                        raw_sentence.append(pred_label)
                        sentence_words.append(display_word)

                        accepted_text = display_word[0].upper() + display_word[1:] if display_word else ''

                        display_hold = DISPLAY_HOLD_FRAMES
                        cooldown = COOLDOWN_FRAMES
                        sequence.clear()
                        pred_history.clear()
                        recent_hands.clear()

                        if display_word != last_spoken_text:
                            speak_text(display_word)
                            last_spoken_text = display_word

                if not any(recent_hands):
                    pred_history.clear()

            display = make_display(image, accepted_text, sentence_words)
            cv2.imshow("Hybrid ASL Realtime", display)

            key = cv2.waitKey(10) & 0xFF

            if key == ord('q'):
                break

            if key == ord('c'):
                sentence_words = []
                raw_sentence = []
                accepted_text = "Waiting..."
                last_spoken_text = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()