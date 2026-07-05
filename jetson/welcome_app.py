import os
import sys
import subprocess

APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_DIR)
import wave
from datetime import datetime

# تنظيف متغيرات النظام لبيئة عمل الـ Qt على جهاز Jetson
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_PLUGIN_PATH", None)

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QCursor
from PyQt5.QtWidgets import QGraphicsDropShadowEffect

# 🟢 حيلة برمجية ذكية لحل مشكلة الـ ONNX Runtime والـ Providers لـ Piper على الـ Jetson
try:
    import onnxruntime as ort
    orig_InferenceSession = ort.InferenceSession
    class PatchedInferenceSession(orig_InferenceSession):
        def __init__(self, path_or_bytes, sess_options=None, providers=None, **kwargs):
            if providers is None:
                providers = ['CPUExecutionProvider']
            super().__init__(path_or_bytes, sess_options=sess_options, providers=providers, **kwargs)
    ort.InferenceSession = PatchedInferenceSession
    print("✅ Successfully patched ONNX Runtime Providers for Welcome Speech.")
except Exception as e:
    print(f"⚠️ Failed to patch ONNX Runtime: {e}")

try:
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    print("⚠️ Piper library not found in main.py environment.")


VOICE_KEYS = ['en_male','ar_male' , 'en_female']
VOICE_TEXTS = {
    'ar_male': ("مرحباً بكم في نظام ساينماتِك لترجمة لغة الإشارة الحية, نَكسر الحواجز لنَبني التواصل.", "voices/kareem.onnx"),
    
    'en_male': (" ...... Welcome to SignMatic, real-time sign language translation system. Bridging the gap, connecting worlds.", "voices/ryan.onnx"),
    
    'en_female': ("Welcome to SignMatic, real-time sign language translation system. Bridging the gap, connecting worlds.", "voices/hfc_female.onnx")
}
# ============================================================

class SignMaticWelcomeUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("SignMatic - Welcome Screen")
        self.setMinimumSize(1000, 700)
        self.showFullScreen()


        self.current_voice_index = 0
        self.is_muted = False
        self.speech_process = None  

        self.setStyleSheet("""
            QMainWindow {
                border-image: url("assets/welcome_background.jpg") 0 0 0 0 stretch stretch;
            }
            QLabel {
                font-family: 'Segoe UI', Arial, sans-serif;
                background-color: transparent;
            }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(40, 50, 40, 60)
        main_layout.setSpacing(20)


        main_layout.addLayout(self.create_top_dashboard())
        
        main_layout.addStretch(1)


        welcome_box = QVBoxLayout()
        welcome_box.setSpacing(10)
        
        title_label = QLabel("WELCOME TO SIGNMATIC")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 42px; font-weight: 900; color: #ffffff; letter-spacing: 3px;")
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(0, 4)
        title_label.setGraphicsEffect(shadow)

        subtitle_label = QLabel("Real-Time Sign Language Translation System")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("font-size: 18px; font-weight: 500; color: rgba(255, 255, 255, 190); font-style: italic;")

        desc_label = QLabel("Breaking barriers, connecting hearts. Press Start to initialize camera and AI models.")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setStyleSheet("font-size: 14px; color: rgba(255, 255, 255, 140); padding-top: 10px;")

        welcome_box.addWidget(title_label)
        welcome_box.addWidget(subtitle_label)
        welcome_box.addWidget(desc_label)
        main_layout.addLayout(welcome_box)

        main_layout.addStretch(1)


        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("🚀   START SYSTEM")
        self.btn_start.setFixedSize(360, 75)
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 25);
                color: #ffffff;
                font-size: 18px;
                font-weight: bold;
                border-radius: 25px;
                border: 2px solid rgba(255, 255, 255, 40);
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background-color: #fbbc05;
                color: #121622;
                border: 2px solid #fbbc05;
            }
            QPushButton:pressed {
                background-color: #d8a004;
                padding-top: 3px;
                padding-left: 3px;
            }
        """)
        
        btn_shadow = QGraphicsDropShadowEffect()
        btn_shadow.setBlurRadius(40)
        btn_shadow.setColor(QColor(0, 0, 0, 120))
        btn_shadow.setOffset(0, 8)
        self.btn_start.setGraphicsEffect(btn_shadow)
        
        self.btn_start.clicked.connect(self.trigger_initialization_state)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_start)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)


        footer_label = QLabel("🤖 Jetson Hardware Stack Integrated  •  Press ESC to exit")
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setStyleSheet("color: rgba(255, 255, 255, 90); font-size: 11px; font-weight: bold;")
        main_layout.addWidget(footer_label)


        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.update_live_dashboard_data)
        self.ui_timer.start(1000)
        self.update_live_dashboard_data()


        self.loop_speech_timer = QTimer(self)
        self.loop_speech_timer.timeout.connect(self.play_next_welcome_speech)
        self.loop_speech_timer.start(9500)
        

        QTimer.singleShot(1000, self.play_next_welcome_speech)

    def create_top_dashboard(self):
        layout = QHBoxLayout()
        
        self.temp_card = QLabel("SYSTEM STATUS\nCPU TEMP: --.-°C 🌡️")
        self.temp_card.setStyleSheet("""
            background-color: rgba(30, 35, 50, 130); color: #ffffff; border-radius: 18px;
            padding: 12px 25px; font-size: 13px; font-weight: bold; border: 1px solid rgba(255, 255, 255, 25);
        """)
        
        self.clock_card = QLabel()
        self.clock_card.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.clock_card.setStyleSheet("""
            background-color: rgba(30, 35, 50, 130); color: #ffffff; border-radius: 18px;
            padding: 12px 25px; font-size: 14px; font-weight: bold; border: 1px solid rgba(255, 255, 255, 25);
        """)


        self.btn_mute = QPushButton("🔈")
        self.btn_mute.setFixedSize(45, 45)
        self.btn_mute.setCursor(Qt.PointingHandCursor)
        self.btn_mute.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 12);
                color: rgba(255, 255, 255, 180);
                border-radius: 15px;
                font-size: 16px;
                border: 1px solid rgba(255, 255, 255, 15);
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 30);
                color: #ffffff;
            }
        """)
        self.btn_mute.clicked.connect(self.toggle_welcome_mute)

        layout.addWidget(self.temp_card)
        layout.addStretch()
        layout.addWidget(self.btn_mute)
        layout.addWidget(self.clock_card)
        return layout

    def toggle_welcome_mute(self):

        self.is_muted = not self.is_muted
        if self.is_muted:
            self.btn_mute.setText("🔇")
            self.btn_mute.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255, 71, 87, 30);
                    color: #ff4757;
                    border-radius: 15px;
                    font-size: 16px;
                    border: 1px solid rgba(255, 71, 87, 50);
                }
            """)

            if self.speech_process and self.speech_process.poll() is None:
                self.speech_process.terminate()
        else:
            self.btn_mute.setText("🔈")
            self.btn_mute.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255, 255, 255, 12);
                    color: rgba(255, 255, 255, 180);
                    border-radius: 15px;
                    font-size: 16px;
                    border: 1px solid rgba(255, 255, 255, 15);
                }
            """)

            self.play_next_welcome_speech()




    def play_next_welcome_speech(self):

        if self.is_muted or not PIPER_AVAILABLE:
            return

        try:

            current_voice_config = VOICE_KEYS[self.current_voice_index]
            text_to_speak, model_path = VOICE_TEXTS[current_voice_config]
            
            if os.path.exists(model_path):
                voice = PiperVoice.load(model_path)
                

                if current_voice_config == 'ar_male':
                    custom_speed = 0.80 
                else:
                    custom_speed = 1.3
                
                config = SynthesisConfig(speaker_id=None, noise_scale=0.667, length_scale=custom_speed, noise_w_scale=1)
                
                output_path = f"welcome_{current_voice_config}.wav"
                wav_file = wave.open(output_path, "wb")
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(voice.config.sample_rate)

                for audio_chunk in voice.synthesize(text_to_speak, syn_config=config):
                    wav_file.writeframes(audio_chunk.audio_int16_bytes)
                wav_file.close()


                if self.speech_process and self.speech_process.poll() is None:
                    self.speech_process.terminate()


                self.speech_process = subprocess.Popen(["aplay", output_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            

            self.current_voice_index = (self.current_voice_index + 1) % len(VOICE_KEYS)
            
        except Exception as e:
            print(f"Loop welcome speech error: {e}")




    def update_live_dashboard_data(self):
        now = datetime.now()
        self.clock_card.setText(f"🕒  {now.strftime('%I:%M:%S %p')}\n📅  {now.strftime('%d %B %Y')}")
        
        try:
            thermal_path = "/sys/class/thermal/thermal_zone0/temp"
            if os.path.exists(thermal_path):
                with open(thermal_path, "r") as f:
                    temp_raw = f.read().strip()
                temp_c = int(temp_raw) / 1000.0
                status_color = "🟢 Stable" if temp_c < 65 else "🟠 Warning"
                self.temp_card.setText(f"🖥️  JETSON HARDWARE: {status_color}\n🌡️  CPU TEMPERATURE: {temp_c:.1f}°C")
            else:
                self.temp_card.setText("🖥️  JETSON HARDWARE: Active\n🌡️  CPU TEMPERATURE: N/A")
        except Exception:
            self.temp_card.setText("🖥️  JETSON HARDWARE: Active\n🌡️  CPU TEMPERATURE: Error")

    def trigger_initialization_state(self):

        self.loop_speech_timer.stop()
        if self.speech_process and self.speech_process.poll() is None:
            self.speech_process.terminate()

        self.btn_start.setEnabled(False) 
        self.btn_start.setText("⌛   INITIALIZING AI MODELS...")
        
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #00d2ff;
                color: #121622;
                font-size: 16px;
                font-weight: bold;
                border-radius: 25px;
                border: 2px solid #00d2ff;
                letter-spacing: 0px;
            }
        """)
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        

        QTimer.singleShot(400, self.launch_main_translator_system)

    def launch_main_translator_system(self):

        target_script = "signmatic_kiosk.py"
        if os.path.exists(target_script):
            subprocess.Popen([sys.executable, target_script])
            QTimer.singleShot(6000, self.safely_finalize_and_close)
        else:
            QApplication.restoreOverrideCursor()
            self.btn_start.setText("❌ ERROR: signmatic_kiosk.py NOT FOUND")

    def safely_finalize_and_close(self):

        QApplication.restoreOverrideCursor() 
        self.close() 

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if self.speech_process and self.speech_process.poll() is None:
                self.speech_process.terminate()
            QApplication.restoreOverrideCursor()
            self.close()
        else:
            super().keyPressEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    window = SignMaticWelcomeUI()
    window.show()
    sys.exit(app.exec_())
