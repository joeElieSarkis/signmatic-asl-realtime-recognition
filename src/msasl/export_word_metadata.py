import json
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'MSASL', 'MSASL_download')
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'MSASL', 'MSASL_selected')
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_JSON = os.path.join(DATA_DIR, 'MSASL_train.json')
VAL_JSON = os.path.join(DATA_DIR, 'MSASL_val.json')
TEST_JSON = os.path.join(DATA_DIR, 'MSASL_test.json')

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
    'thanks': 'Thanks'
}

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

all_items = []
for split_name, path in [('train', TRAIN_JSON), ('val', VAL_JSON), ('test', TEST_JSON)]:
    data = load_json(path)
    for item in data:
        word = item['text'].strip().lower()
        if word in TARGET_WORDS:
            out = dict(item)
            out['split_name'] = split_name
            out['display_label'] = TARGET_WORDS[word]
            all_items.append(out)

out_path = os.path.join(OUT_DIR, 'selected_words_metadata.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(all_items, f, indent=2)

print(f"Saved {len(all_items)} matching samples to:")
print(out_path)

print("\nWords exported:")
for word, label in TARGET_WORDS.items():
    print(f"{word:14s} -> {label}")