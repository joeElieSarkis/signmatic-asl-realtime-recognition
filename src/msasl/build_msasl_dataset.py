import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
KEYPOINTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'MSASL', 'MSASL_keypoints')
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'MSASL', 'processed_msasl')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_LABELS = [
    'Hello',
    'Yes',
    'No',
    'Help',
    'Water',
    'Eat',
    'Fine',
    'Please',
    'Good',
    'Nice'
]

SEQ_LEN = 30
FEATURE_DIM = 258

def fix_sequence_length(seq, target_len=30):
    if len(seq) == 0:
        return np.zeros((target_len, FEATURE_DIM), dtype=np.float32)

    if len(seq) == target_len:
        return seq.astype(np.float32)

    if len(seq) > target_len:
        idxs = np.linspace(0, len(seq) - 1, target_len).astype(int)
        return seq[idxs].astype(np.float32)

    pad_count = target_len - len(seq)
    pad = np.repeat(seq[-1][np.newaxis, :], pad_count, axis=0)
    return np.vstack([seq, pad]).astype(np.float32)

X = []
y = []

label_map = {label: i for i, label in enumerate(TARGET_LABELS)}

for label in TARGET_LABELS:
    label_dir = os.path.join(KEYPOINTS_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    for fname in os.listdir(label_dir):
        if not fname.endswith('.npy'):
            continue

        fpath = os.path.join(label_dir, fname)
        seq = np.load(fpath)

        if seq.ndim != 2 or seq.shape[1] != FEATURE_DIM:
            continue

        seq_fixed = fix_sequence_length(seq, SEQ_LEN)
        X.append(seq_fixed)
        y.append(label_map[label])

X = np.array(X, dtype=np.float32)
y = np.array(y)

y_cat = to_categorical(y, num_classes=len(TARGET_LABELS)).astype(np.float32)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)

y_temp_labels = np.argmax(y_temp, axis=1)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp_labels
)

np.save(os.path.join(OUT_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(OUT_DIR, 'X_val.npy'), X_val)
np.save(os.path.join(OUT_DIR, 'X_test.npy'), X_test)
np.save(os.path.join(OUT_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(OUT_DIR, 'y_val.npy'), y_val)
np.save(os.path.join(OUT_DIR, 'y_test.npy'), y_test)

with open(os.path.join(OUT_DIR, 'labels.txt'), 'w', encoding='utf-8') as f:
    for label in TARGET_LABELS:
        f.write(label + '\n')

print("Dataset built.")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)