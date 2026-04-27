import os
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

CUSTOM_DIR = os.path.join(PROJECT_ROOT, 'data', 'Custom', 'custom_keypoints')
MSASL_DIR = os.path.join(PROJECT_ROOT, 'data', 'MSASL', 'MSASL_keypoints')
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'Hybrid', 'processed_hybrid_25')

os.makedirs(OUT_DIR, exist_ok=True)

WORDS = [
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
    'High',
    'Grade',
    'Lebanese',
    'International',
    'University'
]

CLASSES = WORDS + ['Idle']

MSASL_WORDS = {
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
    'Have',
    'Work',
    'So',
    'Hard',
    'Live',
    'Love',
    'Thanks',
    'University'
}

SEQ_LEN = 30
FEATURE_DIM = 258

CUSTOM_LIMIT_PER_WORD = None
MSASL_LIMIT_PER_WORD = None
IDLE_LIMIT = None

label_map = {label: i for i, label in enumerate(CLASSES)}

def load_custom_sequences(label):
    label_dir = os.path.join(CUSTOM_DIR, label)
    samples = []

    if not os.path.isdir(label_dir):
        return samples

    seq_names = sorted(
        [x for x in os.listdir(label_dir) if os.path.isdir(os.path.join(label_dir, x))],
        key=lambda x: int(x) if x.isdigit() else x
    )

    for seq_name in seq_names:
        seq_dir = os.path.join(label_dir, seq_name)
        frames = []
        ok = True

        for frame_num in range(SEQ_LEN):
            fpath = os.path.join(seq_dir, f'{frame_num}.npy')
            if not os.path.exists(fpath):
                ok = False
                break
            arr = np.load(fpath)
            if arr.shape[0] != FEATURE_DIM:
                ok = False
                break
            frames.append(arr)

        if ok and len(frames) == SEQ_LEN:
            samples.append(np.array(frames, dtype=np.float32))

    return samples

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

def load_msasl_sequences(label):
    label_dir = os.path.join(MSASL_DIR, label)
    samples = []

    if not os.path.isdir(label_dir):
        return samples

    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith('.npy'):
            continue

        fpath = os.path.join(label_dir, fname)
        seq = np.load(fpath)

        if seq.ndim != 2 or seq.shape[1] != FEATURE_DIM:
            continue

        seq_fixed = fix_sequence_length(seq, SEQ_LEN)
        samples.append(seq_fixed)

    return samples

X = []
y = []
source_counts = defaultdict(lambda: {'custom': 0, 'msasl': 0})

for label in WORDS:
    custom_samples = load_custom_sequences(label)

    msasl_samples = []
    if label in MSASL_WORDS:
        msasl_samples = load_msasl_sequences(label)

    if CUSTOM_LIMIT_PER_WORD is not None:
        custom_samples = custom_samples[:CUSTOM_LIMIT_PER_WORD]

    if MSASL_LIMIT_PER_WORD is not None:
        msasl_samples = msasl_samples[:MSASL_LIMIT_PER_WORD]

    for seq in custom_samples:
        X.append(seq)
        y.append(label_map[label])
        source_counts[label]['custom'] += 1

    for seq in msasl_samples:
        X.append(seq)
        y.append(label_map[label])
        source_counts[label]['msasl'] += 1

idle_samples = load_custom_sequences('Idle')
if IDLE_LIMIT is not None:
    idle_samples = idle_samples[:IDLE_LIMIT]

for seq in idle_samples:
    X.append(seq)
    y.append(label_map['Idle'])
    source_counts['Idle']['custom'] += 1

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

print("Hybrid 25-word dataset built.")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)

print("\nPer-class source counts:")
for label in CLASSES:
    c = source_counts[label]
    print(f"{label:14s} custom={c['custom']:4d} msasl={c['msasl']:4d} total={c['custom'] + c['msasl']:4d}")