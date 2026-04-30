import os
import re
import json
import cv2
from urllib.parse import urlparse, parse_qs

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
META_PATH = os.path.join(PROJECT_ROOT, 'data', 'MSASL', 'MSASL_selected', 'selected_words_metadata.json')
VIDEOS_DIR = os.path.join(PROJECT_ROOT, 'data', 'MSASL', 'MSASL_videos')
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'MSASL', 'MSASL_clips')

TARGET_WORDS = {
    'hello': 'Hello',
    'yes': 'Yes',
    'no': 'No',
    'help': 'Help',
    'water': 'Water',
    'eat': 'Eat',
    'fine': 'Fine',
    'please': 'Please',
    'good': 'Good',
    'nice': 'Nice',
    'give': 'Give',
    'we': 'We',
    'have': 'Have',
    'work': 'Work',
    'so': 'So',
    'hard': 'Hard',
    'live': 'Live',
    'love': 'Love',
    'university': 'University',
    'thanks': 'Thanks',
    'teacher': 'Teacher',
    'happy': 'Happy',
    'like': 'Like',
    'want': 'Want',
    'deaf': 'Deaf',
    'school': 'School',
    'what': 'What',
    'need': 'Need',
    'friend': 'Friend',
    'learn': 'Learn',
    'book': 'Book',
    'computer': 'Computer'
}

os.makedirs(OUT_DIR, exist_ok=True)

for label in TARGET_WORDS.values():
    os.makedirs(os.path.join(OUT_DIR, label), exist_ok=True)

def get_youtube_id(url):
    if 'youtube.com' in url:
        q = parse_qs(urlparse(url).query)
        if 'v' in q:
            return q['v'][0]
    if 'youtu.be/' in url:
        return url.rstrip('/').split('/')[-1]
    m = re.search(r'v=([A-Za-z0-9_-]{11})', url)
    return m.group(1) if m else None

def find_video_file(video_id):
    for ext in ['.mp4', '.mkv', '.webm', '.mov']:
        p = os.path.join(VIDEOS_DIR, video_id + ext)
        if os.path.exists(p):
            return p
    return None

with open(META_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

saved = 0
skipped = 0

for idx, item in enumerate(data):
    word = item['text'].strip().lower()

    if word not in TARGET_WORDS:
        continue

    video_id = get_youtube_id(item['url'])
    src = find_video_file(video_id)

    if src is None:
        skipped += 1
        continue

    start_time = float(item['start_time'])
    end_time = float(item['end_time'])
    label = TARGET_WORDS[word]

    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        skipped += 1
        continue

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_name = f"{word}_{video_id}_{idx}.mp4"
    out_path = os.path.join(OUT_DIR, label, out_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current = start_frame

    while current <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        current += 1

    writer.release()
    cap.release()
    saved += 1

print(f"Saved clips: {saved}")
print(f"Skipped: {skipped}")