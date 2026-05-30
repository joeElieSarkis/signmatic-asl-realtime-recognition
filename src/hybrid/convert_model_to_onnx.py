import os
import tensorflow as tf
import tf2onnx

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

KERAS_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'final_signmatic_transformer_50words.h5')
ONNX_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'final_signmatic_transformer_50words.onnx')

model = tf.keras.models.load_model(KERAS_MODEL_PATH)

input_signature = [
    tf.TensorSpec(
        shape=(None, 30, 258),
        dtype=tf.float32,
        name='input'
    )
]

onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13,
    output_path=ONNX_MODEL_PATH
)

print(f"Saved ONNX model to: {ONNX_MODEL_PATH}")