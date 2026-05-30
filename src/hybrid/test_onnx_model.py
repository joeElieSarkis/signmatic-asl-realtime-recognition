import os
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'Hybrid', 'processed_hybrid_50_augmented_v2')
ONNX_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'final_signmatic_transformer_50words.onnx')

X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy')).astype(np.float32)
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

preds = session.run([output_name], {input_name: X_test})[0]

y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(preds, axis=1)

acc = np.mean(y_true == y_pred)

print(f"ONNX test accuracy: {acc:.4f}")
print("Input name:", input_name)
print("Output name:", output_name)