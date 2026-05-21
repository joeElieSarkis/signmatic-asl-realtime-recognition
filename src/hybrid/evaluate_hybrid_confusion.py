import os
import numpy as np
from collections import defaultdict
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'Hybrid', 'processed_hybrid_50_generalized')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_hybrid_model_50words_idle_generalized.h5')

LABELS_PATH = os.path.join(DATA_DIR, 'labels.txt')

X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    CLASSES = [line.strip() for line in f.readlines() if line.strip()]

model = load_model(MODEL_PATH)

probs = model.predict(X_test, verbose=1)

y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(probs, axis=1)

print("\nClassification report:\n")
print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

cm = confusion_matrix(y_true, y_pred)

print("\nPer-class accuracy:\n")
for i, label in enumerate(CLASSES):
    total = np.sum(cm[i])
    correct = cm[i][i]
    acc = correct / total if total > 0 else 0
    print(f"{label:14s} acc={acc:.4f} correct={correct:4d} total={total:4d}")

confusions = []

for true_idx in range(len(CLASSES)):
    for pred_idx in range(len(CLASSES)):
        if true_idx == pred_idx:
            continue

        count = cm[true_idx][pred_idx]

        if count > 0:
            confusions.append(
                (
                    count,
                    CLASSES[true_idx],
                    CLASSES[pred_idx]
                )
            )

confusions.sort(reverse=True, key=lambda x: x[0])

print("\nTop confusion pairs:\n")
for count, true_label, pred_label in confusions[:30]:
    print(f"{true_label:14s} -> {pred_label:14s} count={count}")

print("\nSpecific checks:\n")
TARGET_PAIRS = [
    ('You', 'So'),
    ('So', 'You'),
    ('ILoveYou', 'No'),
    ('ILoveYou', 'Where'),
    ('ILoveYou', 'High'),
    ('What', 'Want'),
    ('Want', 'What'),
    ('Yes', 'Need'),
    ('Need', 'Yes'),
    ('Like', 'My'),
    ('My', 'Like'),
    ('Hard', 'Name'),
    ('Name', 'Hard')
]

for true_label, pred_label in TARGET_PAIRS:
    if true_label in CLASSES and pred_label in CLASSES:
        true_idx = CLASSES.index(true_label)
        pred_idx = CLASSES.index(pred_label)
        print(f"{true_label:14s} -> {pred_label:14s} count={cm[true_idx][pred_idx]}")