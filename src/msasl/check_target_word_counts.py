import json
import os
import re
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'MSASL', 'MSASL_download')
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
    'computer': 'Computer',
    'again': 'Again',
    'father': 'Father',
    'mother': 'Mother',
    'where': 'Where',
    'forget': 'Forget',
    'nothing': 'Nothing',
    'i': 'I',
    'you': 'You',
    'and': 'And',
    'my': 'My',
    'name': 'Name',
    'is': 'Is',
    'iloveyou': 'ILoveYou',
    'i love you': 'ILoveYou',
    'i-love-you': 'ILoveYou'
}

def normalize_word(text):
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    compact = re.sub(r'[^a-z0-9]', '', text)

    if text in TARGET_WORDS:
        return text

    if compact in TARGET_WORDS:
        return compact

    return text

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

train = load_json(TRAIN_JSON)
val = load_json(VAL_JSON)
test = load_json(TEST_JSON)

counts = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0, 'total': 0})

for split_name, data in [('train', train), ('val', val), ('test', test)]:
    for item in data:
        word = normalize_word(item['text'])
        if word in TARGET_WORDS:
            label = TARGET_WORDS[word]
            counts[label][split_name] += 1
            counts[label]['total'] += 1

print("Target MSASL counts:\n")

labels_seen = []
for word, label in TARGET_WORDS.items():
    if label not in labels_seen:
        labels_seen.append(label)

for label in labels_seen:
    c = counts[label]
    print(f"{label:14s} total={c['total']:4d} train={c['train']:4d} val={c['val']:4d} test={c['test']:4d}")