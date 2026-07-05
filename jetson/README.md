# SignMatic Jetson Deployment

This folder contains the Jetson Orin Nano deployment interface used by the final SignMatic prototype.

## Files

- `welcome_app.py` launches the startup screen and then opens the main translator interface.
- `signmatic_kiosk.py` runs the live camera, MediaPipe keypoint extraction, TensorRT inference, sentence logic, bilingual output controls, and Piper speech synthesis.
- `requirements-jetson.txt` lists Python packages used by the Jetson application. TensorRT, CUDA, and PyCUDA must match the JetPack version installed on the Jetson.

## Expected Local Layout

```text
repo-root/
  models/
    final_signmatic_transformer_50words.trt
  jetson/
    welcome_app.py
    signmatic_kiosk.py
    assets/
      welcome_background.jpg
      theme_1.jpg
      theme_2.jpg
      theme_5.jpg
    voices/
      ryan.onnx
      hfc_female.onnx
      kareem.onnx
```

The TensorRT engine and Piper voice models are not tracked in GitHub. They are device assets and should be copied to the Jetson after setup.

## Run

From the repository root on the Jetson:

```bash
cd jetson
python3 welcome_app.py
```

The welcome screen initializes the system and launches `signmatic_kiosk.py`.