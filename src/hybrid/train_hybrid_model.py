import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'Hybrid', 'processed_hybrid_37')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'logs', 'hybrid_37')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

num_classes = y_train.shape[1]

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(30, 258)))
model.add(Dropout(0.3))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

callbacks = [
    TensorBoard(log_dir=LOGS_DIR),
    ModelCheckpoint(
        os.path.join(MODELS_DIR, 'best_hybrid_model_37words_idle.h5'),
        monitor='val_categorical_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=16,
    callbacks=callbacks
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print("Best hybrid 37-word model already saved.")