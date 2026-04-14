"""
Download QuickDraw dataset from Google Cloud Storage.

Fetches raw drawing data for specified classes from the official
QuickDraw dataset repository. Each class is downloaded as a .npy
file containing bitmap representations of user drawings.

Files are truncated to MAX_SAMPLES to limit storage and processing time.
"""
import numpy as np
import os
import requests

# QuickDraw classes to download (5 categories for classification)
CLASSES = ['apple', 'star', 'fork', 'candle', 'eyeglasses']
BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
MAX_SAMPLES = 6000  

os.makedirs('data', exist_ok=True)  

for cl in CLASSES:
    path = f"data/{cl}.npy"
    if os.path.exists(path):
        print(f"[SKIP] {cl} already exists.")
        continue

    url = f"{BASE_URL}{cl.replace(' ', '%20')}.npy"
    print(f"[DOWNLOAD] {cl}...")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(path, 'wb') as f:
            f.write(response.content)

        data = np.load(path)[:MAX_SAMPLES]
        np.save(path, data)
        print(f"[OK] {cl}: {data.shape} saved.")

    except Exception as e:
        print(f"[ERROR] {cl}: {e}")
        if os.path.exists(path):
            os.remove(path)