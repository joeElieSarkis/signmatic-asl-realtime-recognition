import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'Hybrid', 'processed_hybrid_50_augmented_v2')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'logs', 'transformer_50_augmented_v2')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

num_classes = y_train.shape[1]
sequence_length = X_train.shape[1]
feature_dim = X_train.shape[2]

def transformer_encoder(x, head_size, num_heads, ff_dim, dropout):
    attn_output = MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(x, x)

    x = LayerNormalization(epsilon=1e-6)(x + attn_output)

    ff_output = Dense(ff_dim, activation='relu')(x)
    ff_output = Dropout(dropout)(ff_output)
    ff_output = Dense(x.shape[-1])(ff_output)

    x = LayerNormalization(epsilon=1e-6)(x + ff_output)

    return x

inputs = Input(shape=(sequence_length, feature_dim))

x = Dense(128)(inputs)

x = transformer_encoder(
    x,
    head_size=64,
    num_heads=4,
    ff_dim=256,
    dropout=0.25
)

x = transformer_encoder(
    x,
    head_size=64,
    num_heads=4,
    ff_dim=256,
    dropout=0.25
)

x = GlobalAveragePooling1D()(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)

outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

model.summary()

callbacks = [
    TensorBoard(log_dir=LOGS_DIR),
    ModelCheckpoint(
        os.path.join(MODELS_DIR, 'best_transformer_model_50words_augmented_v2.h5'),
        monitor='val_categorical_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True,
        verbose=1
    )
]

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=16,
    callbacks=callbacks
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

print(f"Transformer test loss: {test_loss:.4f}")
print(f"Transformer test accuracy: {test_acc:.4f}")
final_model_path = os.path.join(MODELS_DIR, 'final_signmatic_transformer_50words.h5')
model.save(final_model_path)
print(f"Saved final transformer model to: {final_model_path}")
print("Best transformer model already saved.")