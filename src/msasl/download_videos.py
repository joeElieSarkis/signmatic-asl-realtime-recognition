import json
import os
import yt_dlp

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
META_PATH = os.path.join(PROJECT_ROOT, 'data', 'MSASL', 'MSASL_selected', 'selected_words_metadata.json')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'MSASL', 'MSASL_videos')

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(META_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

ydl_opts = {
    'format': 'mp4',
    'outtmpl': os.path.join(OUTPUT_DIR, '%(id)s.%(ext)s'),
    'quiet': True
}

ydl = yt_dlp.YoutubeDL(ydl_opts)

downloaded = 0
failed = 0

for item in data:
    url = item['url']
    try:
        ydl.download([url])
        downloaded += 1
    except:
        failed += 1

print(f"Downloaded: {downloaded}")
print(f"Failed: {failed}")