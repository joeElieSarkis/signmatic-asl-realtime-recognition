import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'Custom', 'custom_keypoints')
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'Custom', 'processed_custom_20')

os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = [
    'Nice',
    'Eat',
    'Yes',
    'No',
    'Water',
    'Help',
    'Hello',
    'Fine',
    'Good',
    'Please',
    'Give',
    'We',
    'A',
    'Have',
    'Work',
    'So',
    'Hard',
    'Live',
    'Love',
    'Thanks',
    'Idle'
]

SEQUENCE_LENGTH = 30
FEATURE_DIM = 258

label_map = {label: idx for idx, label in enumerate(CLASSES)}

X = []
y = []

for label in CLASSES:
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    for seq_name in sorted(os.listdir(label_dir), key=lambda x: int(x) if x.isdigit() else x):
        seq_dir = os.path.join(label_dir, seq_name)
        if not os.path.isdir(seq_dir):
            continue

        frames = []
        ok = True

        for frame_num in range(SEQUENCE_LENGTH):
            fpath = os.path.join(seq_dir, f'{frame_num}.npy')
            if not os.path.exists(fpath):
                ok = False
                break
            arr = np.load(fpath)
            if arr.shape[0] != FEATURE_DIM:
                ok = False
                break
            frames.append(arr)

        if ok and len(frames) == SEQUENCE_LENGTH:
            X.append(np.array(frames, dtype=np.float32))
            y.append(label_map[label])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
y_cat = to_categorical(y, num_classes=len(CLASSES)).astype(np.float32)

X_train, X_temp, y_train, y_temp, y_train_raw, y_temp_raw = train_test_split(
    X, y_cat, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp_raw
)

np.save(os.path.join(OUT_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(OUT_DIR, 'X_val.npy'), X_val)
np.save(os.path.join(OUT_DIR, 'X_test.npy'), X_test)
np.save(os.path.join(OUT_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(OUT_DIR, 'y_val.npy'), y_val)
np.save(os.path.join(OUT_DIR, 'y_test.npy'), y_test)

with open(os.path.join(OUT_DIR, 'labels.txt'), 'w', encoding='utf-8') as f:
    for label in CLASSES:
        f.write(label + '\n')

print("Custom 20-word dataset built.")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)