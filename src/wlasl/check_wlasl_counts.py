import os

base = os.path.join('data', 'WLASL', 'WLASL_raw')
for cls in sorted(os.listdir(base)):
    p = os.path.join(base, cls)
    if os.path.isdir(p):
        print(cls, len(os.listdir(p)))