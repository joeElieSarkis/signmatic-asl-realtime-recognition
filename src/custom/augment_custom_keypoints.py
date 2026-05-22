import os
import shutil
import random
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

IN_DIR = os.path.join(PROJECT_ROOT, 'data', 'Custom', 'custom_keypoints')
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'Custom', 'custom_keypoints_augmented')

CLASSES = [
    'Nice', 'Eat', 'Yes', 'No', 'Water', 'Help', 'Hello', 'Fine', 'Good', 'Please',
    'Give', 'We', 'A', 'Have', 'Work', 'So', 'Hard', 'Live', 'Love', 'Thanks',
    'High', 'Grade', 'Lebanese', 'International', 'University',
    'Teacher', 'Happy', 'Like', 'Want', 'Deaf', 'School',
    'What', 'Need', 'Friend', 'Learn', 'Book', 'Computer',
    'Again', 'Father', 'Mother', 'Where', 'Forget', 'Nothing',
    'I', 'You', 'And', 'My', 'Name', 'Is', 'ILoveYou',
    'Idle'
]

SEQUENCE_LENGTH = 30
FEATURE_DIM = 258

DEFAULT_AUGMENT_RATIO = 0.10

CLASS_AUGMENT_RATIO = {
    'Yes': 0.15,
    'No': 0.15,
    'Fine': 0.20,
    'Mother': 0.20,
    'You': 0.20,
    'So': 0.05,
    'I': 0.15,
    'Eat': 0.15
}

POSE_OFFSET = 0
LEFT_HAND_OFFSET = 33 * 4
RIGHT_HAND_OFFSET = LEFT_HAND_OFFSET + 21 * 3

def load_sequence(seq_dir):
    frames = []

    for frame_num in range(SEQUENCE_LENGTH):
        fpath = os.path.join(seq_dir, f'{frame_num}.npy')

        if not os.path.exists(fpath):
            return None

        arr = np.load(fpath)

        if arr.shape[0] != FEATURE_DIM:
            return None

        frames.append(arr.astype(np.float32))

    return np.array(frames, dtype=np.float32)

def get_valid_sequence_dirs(label_dir):
    if not os.path.isdir(label_dir):
        return []

    seq_dirs = []

    for name in os.listdir(label_dir):
        full_path = os.path.join(label_dir, name)

        if os.path.isdir(full_path) and name.isdigit():
            seq = load_sequence(full_path)
            if seq is not None:
                seq_dirs.append(full_path)

    seq_dirs.sort(key=lambda p: int(os.path.basename(p)))
    return seq_dirs

def temporal_warp(seq):
    t = seq.shape[0]
    base = np.linspace(0, t - 1, t)

    jitter = np.random.normal(0, 0.35, size=t)
    warped = base + jitter
    warped[0] = 0
    warped[-1] = t - 1
    warped = np.clip(warped, 0, t - 1)
    warped = np.sort(warped)

    out = np.zeros_like(seq)

    for feature_idx in range(seq.shape[1]):
        out[:, feature_idx] = np.interp(base, warped, seq[:, feature_idx])

    return out.astype(np.float32)

def landmark_indices():
    groups = []

    for i in range(33):
        base = POSE_OFFSET + i * 4
        groups.append((base, base + 1, base + 2, base + 3))

    for i in range(21):
        base = LEFT_HAND_OFFSET + i * 3
        groups.append((base, base + 1, base + 2, None))

    for i in range(21):
        base = RIGHT_HAND_OFFSET + i * 3
        groups.append((base, base + 1, base + 2, None))

    return groups

LANDMARK_GROUPS = landmark_indices()

def spatial_augment(seq):
    out = seq.copy()

    scale = random.uniform(0.97, 1.03)
    shift_x = random.uniform(-0.025, 0.025)
    shift_y = random.uniform(-0.025, 0.025)
    noise_std = random.uniform(0.0015, 0.004)

    xs = []
    ys = []

    for x_idx, y_idx, z_idx, v_idx in LANDMARK_GROUPS:
        x_vals = out[:, x_idx]
        y_vals = out[:, y_idx]
        z_vals = out[:, z_idx]

        valid = (np.abs(x_vals) + np.abs(y_vals) + np.abs(z_vals)) > 1e-6

        xs.extend(x_vals[valid].tolist())
        ys.extend(y_vals[valid].tolist())

    if len(xs) == 0 or len(ys) == 0:
        return out.astype(np.float32)

    center_x = float(np.mean(xs))
    center_y = float(np.mean(ys))

    for x_idx, y_idx, z_idx, v_idx in LANDMARK_GROUPS:
        x_vals = out[:, x_idx]
        y_vals = out[:, y_idx]
        z_vals = out[:, z_idx]

        valid = (np.abs(x_vals) + np.abs(y_vals) + np.abs(z_vals)) > 1e-6

        out[valid, x_idx] = ((out[valid, x_idx] - center_x) * scale) + center_x + shift_x
        out[valid, y_idx] = ((out[valid, y_idx] - center_y) * scale) + center_y + shift_y
        out[valid, z_idx] = out[valid, z_idx] * random.uniform(0.98, 1.02)

        out[valid, x_idx] += np.random.normal(0, noise_std, size=np.sum(valid))
        out[valid, y_idx] += np.random.normal(0, noise_std, size=np.sum(valid))
        out[valid, z_idx] += np.random.normal(0, noise_std, size=np.sum(valid))

    return out.astype(np.float32)

def augment_sequence(seq):
    out = seq.copy()

    if random.random() < 0.70:
        out = temporal_warp(out)

    out = spatial_augment(out)

    return out.astype(np.float32)

def save_sequence(seq, out_seq_dir):
    os.makedirs(out_seq_dir, exist_ok=True)

    for frame_num in range(SEQUENCE_LENGTH):
        np.save(os.path.join(out_seq_dir, f'{frame_num}.npy'), seq[frame_num].astype(np.float32))

def main():
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    os.makedirs(OUT_DIR, exist_ok=True)

    total_augmented = 0

    for label in CLASSES:
        label_dir = os.path.join(IN_DIR, label)
        seq_dirs = get_valid_sequence_dirs(label_dir)

        if not seq_dirs:
            print(f"{label:14s} real=   0 augmented=   0")
            continue

        ratio = CLASS_AUGMENT_RATIO.get(label, DEFAULT_AUGMENT_RATIO)
        augment_count = max(1, int(len(seq_dirs) * ratio))

        out_label_dir = os.path.join(OUT_DIR, label)
        os.makedirs(out_label_dir, exist_ok=True)

        selected = random.choices(seq_dirs, k=augment_count)

        for aug_idx, seq_dir in enumerate(selected):
            seq = load_sequence(seq_dir)

            if seq is None:
                continue

            aug_seq = augment_sequence(seq)
            save_sequence(aug_seq, os.path.join(out_label_dir, str(aug_idx)))

            total_augmented += 1

        print(f"{label:14s} real={len(seq_dirs):4d} augmented={augment_count:4d}")

    print("\nDone creating augmented custom keypoints.")
    print(f"Total augmented sequences: {total_augmented}")
    print(f"Output folder: {OUT_DIR}")

if __name__ == '__main__':
    main()