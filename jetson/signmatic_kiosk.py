import os

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_PLUGIN_PATH", None)

import sys
import cv2
import numpy as np
import mediapipe as mp
import subprocess
import re
import wave
from collections import deque
from datetime import datetime

import tensorrt as trt
import pycuda.driver as cuda


try:
    import onnxruntime as ort
    orig_InferenceSession = ort.InferenceSession
    class PatchedInferenceSession(orig_InferenceSession):
        def __init__(self, path_or_bytes, sess_options=None, providers=None, **kwargs):
            if providers is None:
                providers = ['CPUExecutionProvider']
            super().__init__(path_or_bytes, sess_options=sess_options, providers=providers, **kwargs)
    ort.InferenceSession = PatchedInferenceSession
    print("✅ Successfully patched ONNX Runtime Providers for Piper Compatibility.")
except Exception as e:
    print(f"⚠️ Failed to patch ONNX Runtime: {e}")

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QSlider, QFrame, 
                             QProgressBar, QGridLayout, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex, QWaitCondition
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsDropShadowEffect, QGraphicsBlurEffect
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QPainter, QPainterPath


from piper import PiperVoice
from piper.config import SynthesisConfig






# ================= Config =================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_DIR)

PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "final_signmatic_transformer_50words.trt"
)
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

PAST_AFTER_HAVE = {
    'Work': 'worked', 'Want': 'wanted', 'Need': 'needed', 'Learn': 'learned',
    'Love': 'loved', 'Like': 'liked', 'Help': 'helped', 'Eat': 'eaten',
    'Give': 'given', 'Forget': 'forgotten'
}

OBJECT_US_PREVIOUS = {'Give', 'Help'}

DISPLAY_WORDS = {
    'Nice': 'nice', 'Eat': 'eat', 'Yes': 'yes', 'No': 'no', 'Water': 'water',
    'Help': 'help', 'Hello': 'hello', 'Fine': 'fine', 'Good': 'good', 'Please': 'please',
    'Give': 'give', 'We': 'we', 'A': 'A', 'Have': 'have', 'Work': 'work',
    'So': 'so', 'Hard': 'hard', 'Live': 'live', 'Love': 'love', 'Thanks': 'thanks',
    'High': 'high', 'Grade': 'grade', 'Lebanese': 'Lebanese', 'International': 'International',
    'University': 'University', 'Teacher': 'teacher', 'Happy': 'happy', 'Like': 'like',
    'Want': 'want', 'Deaf': 'deaf', 'School': 'school', 'What': 'what', 'Need': 'need',
    'Friend': 'friend', 'Learn': 'learn', 'Book': 'book', 'Computer': 'computer',
    'Again': 'again', 'Father': 'father', 'Mother': 'mother', 'Where': 'where',
    'Forget': 'forget', 'Nothing': 'nothing', 'I': 'I', 'You': 'you', 'And': 'and',
    'My': 'my', 'Name': 'name', 'Is': 'is', 'ILoveYou': 'I love you', 'Idle': ''
}


ARABIC_WORDS = {
    'Nice': 'جَمِيل',
    'Eat': 'آكُل',
    'Yes': 'نَعَم',
    'No': 'لَا',
    'Water': 'مِيَاه',
    'Help': 'مُسَاعَدَة',
    'Hello': 'السَّلَامُ عَلَيْكُمْ',
    'Fine': 'أَنَا بِخَيْر',
    'Good': 'جَيِّد',
    'Please': 'مِنْ فَضْلِكَ',
    'Give': 'أَعْطِنِي',
    'We': 'نَحْنُ',
    'A': 'حَرْف أَلِف',
    'Have': 'أَمْلِك',
    'Work': 'عَمَل',
    'So': 'إِذًا',
    'Hard': 'صَعْب',
    'Live': 'أَعِيش',
    'Love': 'حُبّ',
    'Thanks': 'شُكْرًا',
    'High': 'عَالٍ',
    'Grade': 'دَرَجَة',
    'Lebanese': 'لُبْنَانِيّ',
    'International': 'دَوْلِيّ',
    'University': 'جَامِعَة',
    'Teacher': 'أُسْتَاذ',
    'Happy': 'سَعِيد',
    'Like': 'أُحِبّ',
    'Want': 'أُرِيد',
    'Deaf': 'أَصَمّ',
    'School': 'مَدْرَسَة',
    'What': 'مَاذَا',
    'Need': 'أَحْتَاج',
    'Friend': 'صَدِيق',
    'Learn': 'أَتَعَلَّم',
    'Book': 'كِتَاب',
    'Computer': 'حَاسُوب',
    'Again': 'مَرَّةً أُخْرَى',
    'Father': 'وَالِدِي',
    'Mother': 'وَالِدَتِي',
    'Where': 'أَيْن',
    'Forget': 'أَنْسَى',
    'Nothing': 'لَا شَيْءَ',
    'I': 'أَنَا',
    'You': 'أَنْتَ',
    'And': 'وَ',
    'My': 'الخَاصُّ بِي',
    'Name': 'اسْمِي',
    'Is': 'هُوَ',
    'ILoveYou': 'أَنَا أُحِبُّكَ',
    'Idle': ''
}







BG_COLOR = "#eef0f5"
SHADOW_LIGHT = "#ffffff"
SHADOW_DARK = "#d1d9e6"
TEXT_COLOR = "#333333"
ACCENT_COLOR = "#fbbc05"





# ================= AI Core Functions =================
def speech_word(display_text, lang):
    if lang == 'ar': 
        return display_text
    if display_text == 'live': 
        return 'liv'
    return display_text

def clean_sentence(words_list):
    if not words_list:
        return "Awaiting translation..."
    sentence = " ".join(words_list)
    sentence = re.sub(r'\s+([.,!?])', r'\1', sentence)
    return sentence.strip()






# 🧠 ================= PIPER BACKGROUND VOICE THREAD =================
class VoiceThread(QThread):
    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.queue = deque()
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        
        self.voices_path = {
            ('en', 'male'): "voices/ryan.onnx",
            ('en', 'female'): "voices/hfc_female.onnx",
            ('ar', 'male'): "voices/kareem.onnx",
            ('ar', 'female'): "voices/kareem.onnx"
        }
        
        print("🧠 Piper: Pre-loading Neural Voice Models...")
        self.loaded_voices = {}
        for key, path in self.voices_path.items():
            if os.path.exists(path) and key not in self.loaded_voices:
                try:
                    self.loaded_voices[key] = PiperVoice.load(path)
                    print(f"✅ Loaded {key[0]}_{key[1]} successfully.")
                except Exception as e:
                    print(f"❌ Failed to load {path}: {e}")
        print("✅ Piper Neural Voices Ready")

        self.config = SynthesisConfig(
            speaker_id=None,
            noise_scale=0.667,
            length_scale=1.5,
            noise_w_scale=1
        )

    def queue_speech(self, text, lang, gender):
        self.mutex.lock()
        self.queue.clear()
        self.queue.append((text, lang, gender))
        self.condition.wakeOne()
        self.mutex.unlock()

    def run(self):
        while self._run_flag:
            self.mutex.lock()
            while self._run_flag and len(self.queue) == 0:
                self.condition.wait(self.mutex)
            if not self._run_flag:
                self.mutex.unlock()
                break
            
            text, lang, gender = self.queue.popleft()
            self.mutex.unlock()

            processed_text = speech_word(text, lang)
            
            if lang == 'ar':
                voice = self.loaded_voices.get(('ar', 'male'))
            else:
                voice = self.loaded_voices.get((lang, gender))
                
            if not voice:
                voice = self.loaded_voices.get(('en', 'male'))

            if voice:
                try:
                    output_path = "output.wav"
                    wav_file = wave.open(output_path, "wb")
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(voice.config.sample_rate)

                    for audio_chunk in voice.synthesize(processed_text, syn_config=self.config):
                        wav_file.writeframes(audio_chunk.audio_int16_bytes)
                    wav_file.close()

                    subprocess.run(["aplay", output_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception as e:
                    print(f"Piper runtime speech error: {e}")

    def stop(self):
        self._run_flag = False
        self.mutex.lock()
        self.condition.wakeOne()
        self.mutex.unlock()
        self.wait()





def apply_context(raw_word, raw_sentence):
    previous_raw = raw_sentence[-1] if raw_sentence else None
    if raw_word == 'We' and previous_raw in OBJECT_US_PREVIOUS: return 'us'
    if previous_raw == 'Have' and raw_word in PAST_AFTER_HAVE: return PAST_AFTER_HAVE[raw_word]
    return DISPLAY_WORDS.get(raw_word, raw_word.lower())





class TRTModel:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.input_name = "input"
        self.output_name = "dense_7"
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)
        
        self.d_input = cuda.mem_alloc(trt.volume(self.input_shape) * np.float32().nbytes)
        self.d_output = cuda.mem_alloc(trt.volume(self.output_shape) * np.float32().nbytes)
        self.stream = cuda.Stream()

    def infer(self, input_data):
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()
        return output







# ================= Worker Thread for ML & Camera =================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_data_signal = pyqtSignal(str, str, int, list, str) 

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.raw_sentence = []

    def run(self):
        cuda.init()
        device = cuda.Device(0)
        self.ctx = device.make_context()
        
        print("🚀 Loading TensorRT model...")
        model = TRTModel(MODEL_PATH)
        print("✅ Model loaded")

        sequence = deque(maxlen=SEQUENCE_LENGTH)
        pred_history = deque(maxlen=STABLE_FRAMES)
        recent_hands = deque(maxlen=RECENT_HANDS_FRAMES)

        accepted_text = "Waiting..."
        sentence_words = []
        self.raw_sentence = []

        cooldown = 0
        display_hold = 0

        gst_pipeline = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink drop=1"
        )
        
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self._run_flag and cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                self.draw_landmarks(image, results)

                current_hands = (results.left_hand_landmarks is not None) or (results.right_hand_landmarks is not None)
                recent_hands.append(current_hands)

                kp = self.extract_keypoints(results)
                sequence.append(kp)

                if cooldown > 0: cooldown -= 1
                
                if display_hold > 0:
                    display_hold -= 1
                elif not any(recent_hands):
                    accepted_text = "Waiting..."

                current_conf = 0
                word_to_speak_trigger = "" 
                pred_label = "Idle"

                if len(sequence) == SEQUENCE_LENGTH:
                    input_data = np.expand_dims(np.array(sequence), axis=0).astype(np.float32)
                    output = model.infer(input_data)
                    probs = output[0]

                    pred_idx = int(np.argmax(probs))
                    pred_label = CLASSES[pred_idx]
                    pred_conf = float(probs[pred_idx])
                    current_conf = int(pred_conf * 100)

                    pred_history.append(pred_idx)
                    stable = len(pred_history) == STABLE_FRAMES and len(set(pred_history)) == 1
                    had_recent_hands = any(recent_hands)

                    if stable and cooldown == 0 and had_recent_hands and pred_conf >= CONF_THRESHOLD:
                        if pred_label != 'Idle':
                            display_word = apply_context(pred_label, self.raw_sentence)
                            self.raw_sentence.append(pred_label)
                            sentence_words.append(display_word)

                            accepted_text = display_word[0].upper() + display_word[1:] if display_word else ''
                            display_hold = DISPLAY_HOLD_FRAMES
                            cooldown = COOLDOWN_FRAMES
                            
                            word_to_speak_trigger = pred_label
                            
                            sequence.clear()
                            pred_history.clear()
                            recent_hands.clear()

                    if not any(recent_hands):
                        pred_history.clear()

                raw_pred_label = pred_label if len(self.raw_sentence) > 0 else "Waiting..."
                self.update_data_signal.emit(accepted_text, raw_pred_label, current_conf, list(self.raw_sentence), word_to_speak_trigger)
                self.change_pixmap_signal.emit(image)

        cap.release()
        self.ctx.pop()



    def draw_landmarks(self, image, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                           self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                           self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                           self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                           self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                           self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                           self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([pose, lh, rh]).astype(np.float32)

    def stop(self):
        self._run_flag = False
        self.wait()







# ================= GUI Application =================
class ASLTranslatorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_volume = 35
        self.current_translation_lang = 'en' 
        self.current_voice_lang = 'en'        
        self.current_voice_gender = 'male'    
        
        self.theme_counter = 2 
        
        self.last_word_raw = "Waiting..."
        self.last_sentence_raw_list = []
        self.last_spoken_text = None          

        self.setWindowTitle("ASL Translator - Soft UI")
        self.setMinimumSize(1100, 750)
        self.showFullScreen()
        
        

        #THEME 
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        main_layout.addLayout(self.create_top_bar())
        
        middle_layout = QHBoxLayout()
        middle_layout.setSpacing(20)
        
        self.video_frame = self.create_video_frame()
        middle_layout.addWidget(self.video_frame, stretch=2) 
        
        self.controls_frame = self.create_controls_frame()
        middle_layout.addWidget(self.controls_frame, stretch=1)
        
        main_layout.addLayout(middle_layout)
        
        self.translation_frame = self.create_translation_box()
        main_layout.addWidget(self.translation_frame)
        main_layout.addLayout(self.create_footer())


        self.apply_theme_style()


        self.voice_thread = VoiceThread()
        self.voice_thread.start()


        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_data_signal.connect(self.update_translation_data)
        self.thread.start()

        self.temp_timer = QTimer(self)
        self.temp_timer.timeout.connect(self.update_jetson_temperature)
        self.temp_timer.start(1000)

    def apply_theme_style(self):

        if self.theme_counter == 0:

            bg, shadow_l, shadow_d, text, accent = "#eef0f5", "#ffffff", "#d1d9e6", "#333333", "#fbbc05"
            self.setStyleSheet('QMainWindow { border-image: url("assets/theme_2.jpg") 0 0 0 0 stretch stretch; }')
            
        elif self.theme_counter == 1:

            bg, shadow_l, shadow_d, text, accent = "#121622", "#1d2335", "#0a0d14", "#ffffff", "#00d2ff"
            self.setStyleSheet('QMainWindow { border-image: url("assets/theme_2.jpg") 0 0 0 0 stretch stretch; }')
            
        elif self.theme_counter == 2:

            bg, shadow_l, shadow_d, text, accent = "rgba(20, 24, 35, 200)", "rgba(40, 48, 70, 100)", "rgba(5, 7, 12, 150)", "#ffffff", "#ff4757"
            self.setStyleSheet('QMainWindow { border-image: url("assets/theme_5.jpg") 0 0 0 0 stretch stretch; }')
            
            
        elif self.theme_counter == 3:

            
            bg, shadow_l, shadow_d, text, accent = "#eef0f5", "#ffffff", "#d1d9e6", "#333333", "#fbbc05"
            self.setStyleSheet('QMainWindow { border-image: url("assets/theme_5.jpg") 0 0 0 0 stretch stretch; }')
            
        elif self.theme_counter == 4:

            bg, shadow_l, shadow_d, text, accent = "rgba(20, 24, 35, 200)", "rgba(40, 48, 70, 100)", "rgba(5, 7, 12, 150)", "#ffffff", "#ff4757"
            self.setStyleSheet('QMainWindow { border-image: url("assets/theme_1.jpg") 0 0 0 0 stretch stretch; }')
            
            
        elif self.theme_counter == 5:

            bg, shadow_l, shadow_d, text, accent = "#eef0f5", "#ffffff", "#d1d9e6", "#333333", "#fbbc05"
            self.setStyleSheet('QMainWindow { border-image: url("assets/theme_1.jpg") 0 0 0 0 stretch stretch; }')
            
            


        inner_frame_style = f"""
            background-color: {bg}; border-radius: 25px;
            border-top: 3px solid {shadow_l}; border-left: 3px solid {shadow_l};
            border-bottom: 3px solid {shadow_d}; border-right: 3px solid {shadow_d};
        """
        self.video_frame.setStyleSheet(inner_frame_style)
        self.translation_frame.setStyleSheet(inner_frame_style)
        

        self.current_word_label.setStyleSheet(f"""
            background-color: {bg}; border-radius: 15px; padding: 12px 40px; 
            font-size: 18px; font-weight: bold; color: {accent};
            border: 1px solid {shadow_d};
        """)
        self.trans_text.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {text}; border: none; background: transparent;")
        

        button_style = f"""
            QPushButton {{
                background-color: {bg}; color: {text}; border-radius: 15px; min-height: 45px; font-size: 14px; font-weight: bold;
                border-top: 3px solid {shadow_l}; border-left: 3px solid {shadow_l};
                border-bottom: 3px solid {shadow_d}; border-right: 3px solid {shadow_d};
            }}
            QPushButton:hover {{ color: {accent}; }}
            QPushButton:pressed {{
                border-top: 3px solid {shadow_d}; border-left: 3px solid {shadow_d};
                border-bottom: 3px solid {shadow_l}; border-right: 3px solid {shadow_l};
                color: {accent};
            }}
        """

        for btn in self.findChildren(QPushButton):
            if btn.text() != "⚙" and btn.text() != "🔈" and btn.text() != "🔇" and btn.text() != "➕" and btn.text() != "➖":
                btn.setStyleSheet(button_style)

    def cycle_theme_action(self):
        self.theme_counter = (self.theme_counter + 1) % 6
        self.apply_theme_style()
        
        if self.current_voice_lang == 'ar':
            self.voice_thread.queue_speech(f"   تَمَّ تَغْيِيرُ الْمَظْهَرِ إِلَى النَّمَطِ رَقْمو  {self.theme_counter + 1}", "ar", "male")
        else:
            self.voice_thread.queue_speech(f"Theme profile {self.theme_counter + 1} active", "en", self.current_voice_gender)










    def create_top_bar(self):
        layout = QHBoxLayout()
        logo_label = QLabel("✋ ASL TRANSLATOR")
        logo_label.setStyleSheet("font-size: 22px; font-weight: bold; color: white;")
        
        camera_status = QLabel("🟢 Camera Active")
        camera_status.setAlignment(Qt.AlignCenter)
        camera_status.setStyleSheet(f"""
            background-color: {BG_COLOR}; color: {TEXT_COLOR};
            border-radius: 15px; padding: 5px 15px;
            border-top: 2px solid {SHADOW_DARK}; border-left: 2px solid {SHADOW_DARK};
            border-bottom: 2px solid {SHADOW_LIGHT}; border-right: 2px solid {SHADOW_LIGHT};
        """)
        
        settings_icon = self.create_neu_button("⚙", is_circle=True)
        settings_icon.setFixedSize(40, 40)
        settings_icon.clicked.connect(self.open_system_settings)
        
        self.time_label = QLabel()
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.time_label.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        self.update_time()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)
        
        layout.addWidget(logo_label)
        layout.addStretch()
        layout.addWidget(camera_status)
        layout.addStretch()
        layout.addWidget(settings_icon)
        layout.addWidget(self.time_label)
        return layout
        
        
        
        
        
        
        
    def create_video_frame(self):
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 35, 10, 10) 
        layout.setSpacing(10)
        
        self.current_word_label = QLabel("Output: Waiting...")
        self.current_word_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        self.video_feed_label = QLabel()
        self.video_feed_label.setAlignment(Qt.AlignCenter)
        
    
        self.video_feed_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_feed_label.setStyleSheet("border: none; background: transparent;")

        layout.addWidget(self.current_word_label, alignment=Qt.AlignTop | Qt.AlignCenter)
        layout.addWidget(self.video_feed_label, stretch=1)
        return frame     
        

    def create_sub_glass_box(self, title_text, layout_to_add):
        box = QFrame()
        box.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 12); 
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 20);
            }
        """)
        
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(15, 12, 15, 15)
        box_layout.setSpacing(10)
        
        box_title = QLabel(title_text)
        box_title.setStyleSheet("color: rgba(255, 255, 255, 180); font-size: 11px; font-weight: bold; border: none; background: transparent;")
        box_title.setAlignment(Qt.AlignCenter)
        box_layout.addWidget(box_title)
        
        box_layout.addLayout(layout_to_add)
        return box
        
        
        
        
        
        
        

    def create_controls_frame(self):
        main_container = QWidget()
        
        background_frame = QFrame(main_container)
        background_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(30, 35, 50, 110); 
                border-radius: 25px; 
                border: 1px solid rgba(255, 255, 255, 145); 
            }
        """)
        
        blur_effect = QGraphicsBlurEffect()
        blur_effect.setBlurRadius(5)  
        background_frame.setGraphicsEffect(blur_effect)

        shadow_effect = QGraphicsDropShadowEffect()
        shadow_effect.setBlurRadius(100)          
        shadow_effect.setColor(QColor(0, 0, 0, 80)) 
        shadow_effect.setOffset(0, 10)            
        main_container.setGraphicsEffect(shadow_effect)

        content_widget = QWidget(main_container)
        main_layout = QVBoxLayout(content_widget)
        
        main_layout.setContentsMargins(30, 30, 30, 0)
        main_layout.setSpacing(15) 
        
        title = QLabel("CONTROLS")
        title.setStyleSheet("color: rgba(255, 255, 255, 220); font-size: 13px; font-weight: bold; border: none; background: transparent;")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # 🔊 VOICE LANGUAGE
        grid_voice = QGridLayout()
        grid_voice.setSpacing(12)
        
        btn_voice_arabic = self.create_neu_button("🔊    arabic")
        btn_voice_english = self.create_neu_button("🔊    english")
        
        btn_voice_arabic.clicked.connect(lambda: self.set_voice_language('ar'))
        btn_voice_english.clicked.connect(lambda: self.set_voice_language('en'))
        
        grid_voice.addWidget(btn_voice_arabic, 0, 0)
        grid_voice.addWidget(btn_voice_english, 0, 1)
        
        box_voice = self.create_sub_glass_box("VOICE LANGUAGE", grid_voice)
        main_layout.addWidget(box_voice)

        # 👨🏼 VOICE GENDER
        grid_gender = QGridLayout()
        grid_gender.setSpacing(12)
        
        btn_gender_male = self.create_neu_button("👨🏼    Male") 
        btn_gender_female = self.create_neu_button("👩🏼    Female")
        
        btn_gender_male.clicked.connect(lambda: self.set_voice_gender('male'))
        btn_gender_female.clicked.connect(lambda: self.set_voice_gender('female'))
        
        grid_gender.addWidget(btn_gender_male, 0, 0)
        grid_gender.addWidget(btn_gender_female, 0, 1)
        
        box_gender = self.create_sub_glass_box("VOICE GENDER", grid_gender)
        main_layout.addWidget(box_gender)
        
        # 🇯🇴 TRANSLATE
        grid_translate = QGridLayout()
        grid_translate.setSpacing(12)

        btn_trans_arabic = self.create_neu_button("🇯🇴    arabic") 
        btn_trans_english = self.create_neu_button("🇱🇷    english")
        
        btn_trans_arabic.clicked.connect(lambda: self.set_translation_language('ar'))
        btn_trans_english.clicked.connect(lambda: self.set_translation_language('en'))
        
        grid_translate.addWidget(btn_trans_arabic, 0, 0)
        grid_translate.addWidget(btn_trans_english, 0, 1)
        
        box_translate = self.create_sub_glass_box("TRANSLATE", grid_translate)
        main_layout.addWidget(box_translate)

        # 🔊 VOLUME
        volume_title = QLabel("🔊 Volume")
        volume_title.setAlignment(Qt.AlignCenter)
        volume_title.setStyleSheet("font-size:15px; font-weight:bold; color: rgba(255, 255, 255, 200); border: none; background: transparent;")
        
        volume_layout = QHBoxLayout()
        volume_layout.setContentsMargins(15, 15, 15, 15)

        self.btn_mute = self.create_neu_button("🔈", is_circle=True)
        self.btn_mute.setFixedSize(40, 40)
        self.btn_mute.setCheckable(True)
        self.btn_mute.toggled.connect(self.toggle_mute_state)

        btn_vol_down = self.create_neu_button("➖", is_circle=True)
        btn_vol_down.setFixedSize(40, 40)
        btn_vol_down.clicked.connect(self.volume_down)
        
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(35)
        self.vol_slider.valueChanged.connect(self.set_system_volume)
        self.vol_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border-radius: 5px; height: 10px; background: rgba(18, 22, 34, 150);
                border-top: 1px solid {SHADOW_DARK}; border-left: 1px solid {SHADOW_DARK};
                border-bottom: 1px solid {SHADOW_LIGHT}; border-right: 1px solid {SHADOW_LIGHT};
            }}
            QSlider::sub-page:horizontal {{ background: {ACCENT_COLOR}; border-radius: 5px; }}
            QSlider::handle:horizontal {{
                background: #ffffff; width: 18px; margin: -4px 0;
                border-radius: 9px; border: 2px solid {ACCENT_COLOR};
            }}
        """)
        
        btn_vol_up = self.create_neu_button("➕", is_circle=True)
        btn_vol_up.setFixedSize(40, 40)
        btn_vol_up.clicked.connect(self.volume_up)
        
        volume_layout.addWidget(self.btn_mute)
        volume_layout.addWidget(btn_vol_down)
        volume_layout.addWidget(self.vol_slider)
        volume_layout.addWidget(btn_vol_up)
        
        vol_frame = QFrame()
        vol_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 15); 
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 20);
            }
        """)
        v_lay = QVBoxLayout(vol_frame)
        v_lay.addWidget(volume_title) 
        v_lay.addLayout(volume_layout)
        main_layout.addWidget(vol_frame)
        
        
        

        bottom_grid = QGridLayout()
        bottom_grid.setSpacing(15)
        
        btn_clear = self.create_neu_button("🧹  Clear Sentence")
        btn_replay = self.create_neu_button("🔁  Replay Text")
        

        self.btn_theme_cycle = self.create_neu_button("🌓  Switch Theme (4-in-1)")
        
        btn_clear.clicked.connect(self.clear_current_sentence)
        btn_replay.clicked.connect(self.replay_full_sentence)
        self.btn_theme_cycle.clicked.connect(self.cycle_theme_action)
        
        bottom_grid.addWidget(btn_clear, 0, 0)
        bottom_grid.addWidget(btn_replay, 0, 1)
        bottom_grid.addWidget(self.btn_theme_cycle, 1, 0, 1, 2) 
        
        main_layout.addLayout(bottom_grid)
        main_layout.addStretch()

        layout_binder = QGridLayout(main_container)
        layout_binder.setContentsMargins(10, 10, 10, 10) 
        layout_binder.addWidget(background_frame, 0, 0)
        layout_binder.addWidget(content_widget, 0, 0)
        
        return main_container









    def create_translation_box(self):
        frame = QFrame()
        # 🟢 تحديد سياسة حجم ثابتة عمودياً لكي لا يكبر البوكس تلقائياً ويأخذ مساحة الشاشة
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        frame.setFixedHeight(100) # 🟢 يمكنك تصغير هذا الرقم (مثلاً 100) لجعل البوكس أنحف وأجمل
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(0)
        
        self.trans_text = QLabel("Awaiting translation...")
        # توسيط النص مئة بالمئة عمودياً وأفقياً داخلياً
        self.trans_text.setAlignment(Qt.AlignCenter)
        self.trans_text.setStyleSheet("font-size: 28px; font-weight: bold; color: inherit; border: none; background: transparent;")
        
        layout.addWidget(self.trans_text, stretch=1)
        return frame





    def create_footer(self):
        layout = QHBoxLayout()
        model_label = QLabel("🧠 Model: Transformer 50-Words")
        model_label.setStyleSheet("color: white; font-size: 13px; font-weight: bold;")
        
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Confidence")
        conf_label.setStyleSheet("color: white; font-size: 13px; font-weight: bold;")
        
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedSize(150, 10)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: {BG_COLOR}; border-radius: 5px;
                border-top: 1px solid {SHADOW_DARK}; border-left: 1px solid {SHADOW_DARK};
                border-bottom: 1px solid {SHADOW_LIGHT}; border-right: 1px solid {SHADOW_LIGHT};
            }}
            QProgressBar::chunk {{ background-color: {ACCENT_COLOR}; border-radius: 4px; }}
        """)
        
        self.conf_val = QLabel("0%")
        self.conf_val.setStyleSheet("color: white; font-size: 13px; font-weight: bold;")
        
        conf_layout.addStretch()
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.progress)
        conf_layout.addWidget(self.conf_val)
        conf_layout.addStretch()
        
        
        
        
        
        

        self.jetson_temp_label = QLabel("CPU: --.-°C 🌡️")
        self.jetson_temp_label.setStyleSheet("color: white; font-size: 13px; font-weight: bold;")
        
        layout.addWidget(model_label)
        layout.addLayout(conf_layout)
        layout.addWidget(self.jetson_temp_label)
        return layout










    # ================= Slots =================
    def update_image(self, cv_img):
        try:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            target_w = self.video_feed_label.width()
            target_h = self.video_feed_label.height()
            
            if target_w > 0 and target_h > 0:

                scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                    target_w, target_h, 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                
                masked_pixmap = QPixmap(scaled_pixmap.size())
                masked_pixmap.fill(Qt.transparent)
                
                painter = QPainter(masked_pixmap)
                painter.setRenderHint(QPainter.Antialiasing)
                path = QPainterPath()
                path.addRoundedRect(0, 0, scaled_pixmap.width(), scaled_pixmap.height(), 22, 22)
                
                painter.setClipPath(path)
                painter.drawPixmap(0, 0, scaled_pixmap)
                painter.end()
                
                self.video_feed_label.setPixmap(masked_pixmap)
        except Exception as e:
            print(f"Error updating video feed: {e}")
              

    def update_translation_data(self, word, raw_word, confidence, raw_sentence_list, word_to_speak_trigger):
        if raw_word != "Waiting..." and raw_word != "Idle":
            self.last_word_raw = raw_word
        self.last_sentence_raw_list = raw_sentence_list
            
        if self.current_translation_lang == 'ar':
            ar_word = ARABIC_WORDS.get(raw_word, raw_word)
            self.current_word_label.setText(f"الترجمة: {ar_word}" if raw_word != "Waiting..." else "الترجمة: في الانتظار...")
            
            if self.last_sentence_raw_list:
                ar_sentence_words = [ARABIC_WORDS.get(w, w) for w in self.last_sentence_raw_list]
                self.trans_text.setText(' '.join(ar_sentence_words[-DISPLAY_SENTENCE_WORDS:]))
            else:
                self.trans_text.setText("في انتظار الإشارة...")
        else:
            self.current_word_label.setText(f"Output: {word}")
            
            if self.last_sentence_raw_list:
                processed_words = [apply_context(w, self.last_sentence_raw_list[:i]) for i, w in enumerate(self.last_sentence_raw_list)]
                self.trans_text.setText(clean_sentence(processed_words[-DISPLAY_SENTENCE_WORDS:]))
            else:
                self.trans_text.setText("Awaiting translation...")
        
        if word_to_speak_trigger and word_to_speak_trigger != "Waiting..." and word_to_speak_trigger != 'Idle':
            if self.current_voice_lang == 'ar':
                text_to_speak = ARABIC_WORDS.get(word_to_speak_trigger, word_to_speak_trigger)
            else:
                text_to_speak = apply_context(word_to_speak_trigger, self.last_sentence_raw_list[:-1] if self.last_sentence_raw_list else [])

            if text_to_speak != self.last_spoken_text:
                self.last_spoken_text = text_to_speak
                self.voice_thread.queue_speech(text_to_speak, self.current_voice_lang, self.current_voice_gender)

        self.progress.setValue(confidence)
        self.conf_val.setText(f"{confidence}%")




    def clear_current_sentence(self):
        if hasattr(self, 'thread'):
            self.thread.raw_sentence.clear()
        self.last_sentence_raw_list.clear()
        self.trans_text.setText("Awaiting translation..." if self.current_translation_lang == 'en' else "في انتظار الإشارة...")
        if self.current_voice_lang == 'ar':
            self.voice_thread.queue_speech(    "تَمَّتْ إِزَالَةُ جَمِيعِ السِّجِلَّات", "ar", "male")
        else:
            self.voice_thread.queue_speech("Sentence cleared", "en", self.current_voice_gender)

    def replay_full_sentence(self):
        if not self.last_sentence_raw_list:
            return
            
        if self.current_voice_lang == 'ar':
            full_text = " ".join([ARABIC_WORDS.get(w, w) for w in self.last_sentence_raw_list])
        else:
            processed_words = [apply_context(w, self.last_sentence_raw_list[:i]) for i, w in enumerate(self.last_sentence_raw_list)]
            full_text = clean_sentence(processed_words)
            
        self.voice_thread.queue_speech(full_text, self.current_voice_lang, self.current_voice_gender)







    
    def update_jetson_temperature(self):
        try:
            thermal_path = "/sys/class/thermal/thermal_zone0/temp"
            if os.path.exists(thermal_path):
                with open(thermal_path, "r") as f:
                    temp_raw = f.read().strip()
                temp_c = int(temp_raw) / 1000.0
                self.jetson_temp_label.setText(f"CPU: {temp_c:.1f}°C 🌡️")
            else:
                self.jetson_temp_label.setText("CPU: N/A 🌡️")
        except Exception:
            self.jetson_temp_label.setText("CPU: Err 🌡️")








    def set_translation_language(self, lang):
        self.current_translation_lang = lang
        self.update_translation_data(self.last_word_raw, self.last_word_raw, self.progress.value(), self.last_sentence_raw_list, "")

    def set_voice_language(self, lang):
        self.current_voice_lang = lang
        if lang == 'ar':
            self.voice_thread.queue_speech("تَمَّ اخْتِيَارُ اللُّغَةِ الْعَرَبِيَّة", "ar", "male")
        else:
            self.voice_thread.queue_speech("Voice language changed to English", "en", self.current_voice_gender)

    def set_voice_gender(self, gender):
        if self.current_voice_lang == 'ar' and gender == 'female':
            self.voice_thread.queue_speech("حَالِيًّا، اللُّغَةُ الْعَرَبِيَّةُ تَدْعَمُ صَوْتَ كَرِيمٍ فَقَط  ", "ar", "male")
        
  
            return
            
        self.current_voice_gender = gender
        if self.current_voice_lang == 'ar':
            msg = "تَمَّ تَفْعِيلُ الصَّوْتِ الْأُنْثَوِي" if gender == 'female'  else "تَمَّ تَفْعِيلُ الصَّوْتِ الذُّكُورِي"
            self.voice_thread.queue_speech(msg, "ar", "male")
        else:
            msg = "Gender profile changed to female" if gender == 'female' else "Gender profile changed to male"
            self.voice_thread.queue_speech(msg, "en", gender)
            
            
            
            
            
            
            
            

    # ================= OS & System Functions =================
    def set_system_volume(self, value):
        try:
            subprocess.run(["pactl","set-sink-volume","@DEFAULT_SINK@",f"{value}%"])
            subprocess.run(["pactl","set-sink-mute","@DEFAULT_SINK@","false"])
        except Exception: pass

    def volume_up(self):
        self.vol_slider.setValue(min(self.vol_slider.value()+5,100))

    def volume_down(self):
        self.vol_slider.setValue(max(self.vol_slider.value()-5,0))

    def load_current_volume(self):
        try:
            result = subprocess.check_output(["pactl","get-sink-volume","@DEFAULT_SINK@"]).decode()
            m = re.search(r"(\d+)%", result)
            if m: self.vol_slider.setValue(int(m.group(1)))
        except Exception: pass

    def open_system_settings(self):
        try:
            subprocess.Popen(["gnome-control-center"])
        except Exception: pass

    def update_time(self):
        now = datetime.now()
        self.time_label.setText(f"{now.strftime('%I:%M:%S %p')}\n{now.strftime('%d %B %Y')}")

    def toggle_mute_state(self, checked):
        try:
            if checked:
                self.btn_mute.setText("🔇")
                subprocess.run(["pactl","set-sink-mute","@DEFAULT_SINK@", "true"])
            else:
                self.btn_mute.setText("🔈")
                subprocess.run(["pactl","set-sink-mute","@DEFAULT_SINK@", "false"])
        except Exception: pass

    def create_neu_button(self, text, is_circle=False):
        btn = QPushButton(text)
        radius = "20px" if is_circle else "15px"
        min_height = "40px" if is_circle else "60px"
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_COLOR}; color: {TEXT_COLOR};
                border-radius: {radius}; min-height: {min_height}; font-size: 15px; font-weight: bold;
                border-top: 3px solid {SHADOW_LIGHT}; border-left: 3px solid {SHADOW_LIGHT};
                border-bottom: 3px solid {SHADOW_DARK}; border-right: 3px solid {SHADOW_DARK};
            }}
            QPushButton:hover {{ color: {ACCENT_COLOR}; }}
            QPushButton:pressed {{
                border-top: 3px solid {SHADOW_DARK}; border-left: 3px solid {SHADOW_DARK};
                border-bottom: 3px solid {SHADOW_LIGHT}; border-right: 3px solid {SHADOW_LIGHT};
                color: {ACCENT_COLOR};
            }}
        """)
        return btn

    def closeEvent(self, event):
        self.thread.stop()
        self.voice_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    window = ASLTranslatorUI()
    window.show()
    sys.exit(app.exec_())
