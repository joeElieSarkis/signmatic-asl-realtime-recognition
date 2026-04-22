import os
import re
import json
from collections import defaultdict
from urllib.parse import urlparse, parse_qs

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
META_PATH = os.path.join(PROJECT_ROOT, 'data', 'MSASL', 'MSASL_selected', 'selected_words_metadata.json')
VIDEOS_DIR = os.path.join(PROJECT_ROOT, 'data', 'MSASL', 'MSASL_videos')

def get_youtube_id(url):
    if 'youtube.com' in url:
        q = parse_qs(urlparse(url).query)
        if 'v' in q:
            return q['v'][0]
    if 'youtu.be/' in url:
        return url.rstrip('/').split('/')[-1]
    m = re.search(r'v=([A-Za-z0-9_-]{11})', url)
    return m.group(1) if m else None

video_ids_present = set()
for fn in os.listdir(VIDEOS_DIR):
    base, ext = os.path.splitext(fn)
    if ext.lower() in {'.mp4', '.mkv', '.webm', '.m4a', '.mov'}:
        video_ids_present.add(base)

with open(META_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

counts = defaultdict(int)

for item in data:
    vid = get_youtube_id(item['url'])
    word = item['text'].strip().lower()
    if vid in video_ids_present:
        counts[word] += 1

print("Downloaded counts by word:\n")
for word, c in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{word:14s} {c}")