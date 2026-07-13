# SignMatic Jetson Deployment

This directory contains the Jetson Orin Nano deployment software used by the final SignMatic embedded station. The application performs CSI camera capture, MediaPipe keypoint extraction, TensorRT inference, sentence construction, bilingual output, and offline speech synthesis.

## Deployment Files

- `welcome_app.py` displays the startup interface, system status, CPU temperature, date, and time before launching the main translator.
- `signmatic_kiosk.py` runs the live camera pipeline, MediaPipe extraction, TensorRT inference, real-time acceptance logic, touchscreen controls, English/Arabic output, and Piper speech synthesis.
- `requirements-jetson.txt` lists the Python packages required by the application.

TensorRT, CUDA, and PyCUDA are supplied through NVIDIA JetPack and must match the JetPack version installed on the Jetson.

## Runtime Pipeline

```text
Raspberry Pi Camera Module V2
    -> GStreamer and nvarguscamerasrc
    -> OpenCV frame acquisition
    -> MediaPipe pose and hand landmarks
    -> 30-frame x 258-feature buffer
    -> TensorRT Transformer inference
    -> confidence and stability checks
    -> sentence construction
    -> English/Arabic text
    -> offline Piper speech
```

## Expected Local Layout

```text
repo-root/
  models/
    final_signmatic_transformer_50words.trt

  jetson/
    welcome_app.py
    signmatic_kiosk.py
    requirements-jetson.txt

    assets/
      welcome_background.jpg
      theme_1.jpg
      theme_2.jpg
      theme_5.jpg

    voices/
      ryan.onnx
      ryan.onnx.json
      hfc_female.onnx
      hfc_female.onnx.json
      kareem.onnx
      kareem.onnx.json
```

The filenames shown above match the paths used by the deployment code.

The `assets/` and `voices/` directories are retained in Git with `.gitkeep` files. Runtime interface images, Piper voice models, and voice configuration files remain local because they are downloaded, device-specific, or subject to separate distribution terms.

The interface background images can be replaced with compatible local images while preserving the expected filenames.

## Model Requirements

The application loads the TensorRT engine from:

```text
models/final_signmatic_transformer_50words.trt
```

The engine must accept an input sequence shaped as:

```text
1 x 30 x 258
```

The output must contain 51 class probabilities in the class order defined in `signmatic_kiosk.py`.

## Python Environment

Create a virtual environment that can access JetPack-provided system packages:

```bash
cd <repo-root>
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r jetson/requirements-jetson.txt
```

TensorRT and PyCUDA should be installed through the JetPack-compatible NVIDIA packages rather than replaced by unrelated Python wheels.

## Camera Configuration

The live interface uses the Jetson CSI camera pipeline through `nvarguscamerasrc`. The deployed station uses a Raspberry Pi Camera Module V2 connected to the Jetson camera interface.

The application configures camera acquisition through GStreamer and opens the resulting stream with OpenCV.

## Voice Profiles

The deployment uses the following offline Piper voice models:

- `ryan.onnx`: English male voice.
- `hfc_female.onnx`: English female voice presented by the interface as Amy.
- `kareem.onnx`: Arabic male voice.

Arabic female speech falls back to the available Arabic Kareem profile.

Each `.onnx` voice model requires its corresponding `.onnx.json` configuration file in the same directory.

## Real-Time Acceptance Logic

The TensorRT output is filtered before a word is accepted:

- Minimum confidence: `0.88`.
- Required stable predictions: 5 consecutive frames.
- Recent hand-presence check: 8-frame history.
- Cooldown after an accepted word: 6 frames.
- Idle-state handling to prevent unsupported or inactive frames from becoming sentence tokens.
- Displayed sentence length: up to 10 words.

These checks reduce repeated words and prevent isolated high-probability frames from being accepted immediately.

## Interface Functions

The touchscreen application provides:

- Live camera preview with pose and hand landmarks.
- Recognized-word and confidence display.
- English and Arabic text output.
- English male and female voice profiles.
- Arabic male voice profile.
- Sentence clearing and replay.
- Volume and mute controls.
- Theme switching.
- Date, time, and Jetson CPU-temperature display.
- Local system-settings access.

## Run the Application

From the repository root:

```bash
cd jetson
python3 welcome_app.py
```

The welcome interface initializes first. Selecting **Start System** launches `signmatic_kiosk.py`.

The main translator can also be launched directly:

```bash
cd jetson
python3 signmatic_kiosk.py
```

## Local-Only Artifacts

The following runtime artifacts are intentionally excluded from Git:

- TensorRT engines.
- ONNX and Keras model files.
- Piper `.onnx` voice models.
- Piper `.onnx.json` voice configuration files.
- Interface background and theme images.
- Generated `.wav` speech files.
- Device logs and temporary runtime output.

After these assets are installed, recognition, text translation, and speech synthesis operate locally without requiring an internet connection.