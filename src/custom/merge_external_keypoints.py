import os
import shutil
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

PARTNER_DIRS = [
    os.path.join(PROJECT_ROOT, 'data', 'Partner', 'part1'),
    os.path.join(PROJECT_ROOT, 'data', 'Partner', 'part2')
]

CUSTOM_DIR = os.path.join(PROJECT_ROOT, 'data', 'Custom', 'custom_keypoints')

SEQUENCE_LENGTH = 30
FEATURE_DIM = 258

VALID_CLASSES = [
    'Nice', 'Eat', 'Yes', 'No', 'Water', 'Help', 'Hello', 'Fine', 'Good', 'Please',
    'Give', 'We', 'A', 'Have', 'Work', 'So', 'Hard', 'Live', 'Love', 'Thanks',
    'High', 'Grade', 'Lebanese', 'International', 'University',
    'Teacher', 'Happy', 'Like', 'Want', 'Deaf', 'School',
    'What', 'Need', 'Friend', 'Learn', 'Book', 'Computer',
    'Again', 'Father', 'Mother', 'Where', 'Forget', 'Nothing',
    'I', 'You', 'And', 'My', 'Name', 'Is', 'ILoveYou',
    'Idle'
]

CLASS_MAP = {label.lower(): label for label in VALID_CLASSES}

def get_next_index(label_dir):
    if not os.path.isdir(label_dir):
        return 0

    existing = [
        d for d in os.listdir(label_dir)
        if os.path.isdir(os.path.join(label_dir, d)) and d.isdigit()
    ]

    if not existing:
        return 0

    return max(int(x) for x in existing) + 1

def sequence_is_valid(seq_dir):
    for frame_num in range(SEQUENCE_LENGTH):
        fpath = os.path.join(seq_dir, f'{frame_num}.npy')

        if not os.path.exists(fpath):
            return False

        try:
            arr = np.load(fpath)
        except:
            return False

        if arr.shape[0] != FEATURE_DIM:
            return False

    return True

def copy_sequence(src_seq_dir, dst_seq_dir):
    os.makedirs(dst_seq_dir, exist_ok=True)

    for frame_num in range(SEQUENCE_LENGTH):
        src = os.path.join(src_seq_dir, f'{frame_num}.npy')
        dst = os.path.join(dst_seq_dir, f'{frame_num}.npy')
        shutil.copy2(src, dst)

def main():
    os.makedirs(CUSTOM_DIR, exist_ok=True)

    total_copied = 0
    total_skipped = 0

    for partner_dir in PARTNER_DIRS:
        if not os.path.isdir(partner_dir):
            print(f"Missing partner folder: {partner_dir}")
            continue

        for raw_label in sorted(os.listdir(partner_dir)):
            src_label_dir = os.path.join(partner_dir, raw_label)

            if not os.path.isdir(src_label_dir):
                continue

            label_key = raw_label.strip().lower()

            if label_key not in CLASS_MAP:
                print(f"Skipping unknown class: {raw_label}")
                total_skipped += 1
                continue

            label = CLASS_MAP[label_key]
            dst_label_dir = os.path.join(CUSTOM_DIR, label)
            os.makedirs(dst_label_dir, exist_ok=True)

            next_idx = get_next_index(dst_label_dir)

            seq_names = sorted(
                [
                    d for d in os.listdir(src_label_dir)
                    if os.path.isdir(os.path.join(src_label_dir, d)) and d.isdigit()
                ],
                key=lambda x: int(x)
            )

            copied_for_label = 0
            skipped_for_label = 0

            for seq_name in seq_names:
                src_seq_dir = os.path.join(src_label_dir, seq_name)

                if not sequence_is_valid(src_seq_dir):
                    skipped_for_label += 1
                    total_skipped += 1
                    continue

                dst_seq_dir = os.path.join(dst_label_dir, str(next_idx))
                copy_sequence(src_seq_dir, dst_seq_dir)

                next_idx += 1
                copied_for_label += 1
                total_copied += 1

            print(f"{label:14s} copied={copied_for_label:4d} skipped={skipped_for_label:4d}")

    print("\nDone merging partner keypoints.")
    print(f"Total copied: {total_copied}")
    print(f"Total skipped: {total_skipped}")

if __name__ == '__main__':
    main()