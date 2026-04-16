import json
import os
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'MSASL', 'MSASL_download')
TRAIN_JSON = os.path.join(DATA_DIR, 'MSASL_train.json')
VAL_JSON = os.path.join(DATA_DIR, 'MSASL_val.json')
TEST_JSON = os.path.join(DATA_DIR, 'MSASL_test.json')

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

train = load_json(TRAIN_JSON)
val = load_json(VAL_JSON)
test = load_json(TEST_JSON)

counts = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0, 'total': 0})

for item in train:
    word = item['text'].strip()
    counts[word]['train'] += 1
    counts[word]['total'] += 1

for item in val:
    word = item['text'].strip()
    counts[word]['val'] += 1
    counts[word]['total'] += 1

for item in test:
    word = item['text'].strip()
    counts[word]['test'] += 1
    counts[word]['total'] += 1

sorted_counts = sorted(counts.items(), key=lambda x: x[1]['total'], reverse=True)

print(f"Unique words: {len(sorted_counts)}")
print("\nTop 50 words by total samples:\n")
for word, c in sorted_counts[:50]:
    print(f"{word:20s} total={c['total']:4d} train={c['train']:4d} val={c['val']:4d} test={c['test']:4d}")