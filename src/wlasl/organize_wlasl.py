import os
import json
import shutil

# ===== PATHS =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VIDEOS_DIR = os.path.join(PROJECT_ROOT, 'data', 'WLASL', 'WLASL_download', 'videos')  
JSON_PATH = os.path.join(PROJECT_ROOT, 'data', 'WLASL', 'WLASL_download', 'WLASL_v0.3.json')  
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'WLASL', 'WLASL_raw')

# ===== PICK YOUR WORDS =====
TARGET_GLOSSES = {
    'go': 'Go',
    'fine': 'Fine',
    'help': 'Help',
    'no': 'No',
    'water': 'Water',
    'yes': 'Yes'
}

VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.mkv']

def find_video_file(video_id: str):
    candidates = [video_id]
    if video_id.isdigit():
        candidates.append(video_id.zfill(5))

    for cand in candidates:
        for ext in VIDEO_EXTS:
            path = os.path.join(VIDEOS_DIR, cand + ext)
            if os.path.exists(path):
                return path
    return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for display_name in TARGET_GLOSSES.values():
        os.makedirs(os.path.join(OUTPUT_DIR, display_name), exist_ok=True)

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    copied = 0
    missing = []

    for entry in data:
        gloss = entry['gloss'].strip().lower()

        if gloss not in TARGET_GLOSSES:
            continue

        display_label = TARGET_GLOSSES[gloss]
        out_class_dir = os.path.join(OUTPUT_DIR, display_label)

        for idx, inst in enumerate(entry['instances']):
            video_id = str(inst['video_id'])
            src_video = find_video_file(video_id)

            if src_video is None:
                missing.append((gloss, video_id))
                continue

            split = inst.get('split', 'unknown')
            dst_name = f"{gloss.replace(' ', '_')}_{split}_{video_id}_{idx}.mp4"
            dst_path = os.path.join(out_class_dir, dst_name)

            if not os.path.exists(dst_path):
                shutil.copy2(src_video, dst_path)
                copied += 1

    print(f"Copied videos: {copied}")
    print(f"Missing videos: {len(missing)}")

    if missing:
        print("\nFirst 20 missing:")
        for item in missing[:20]:
            print(item)

if __name__ == "__main__":
    main()