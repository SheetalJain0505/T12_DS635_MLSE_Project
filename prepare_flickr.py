"""
prepare_flickr.py (FINAL VERSION FOR YOUR SYSTEM)

Creates flickr_data/corpus.json from your Flickr8k dataset folder structure.

Your actual folder layout:
- Flickr8k_Dataset/Flicker8k_Dataset/*.jpg   (images)
- Flickr8k_text/Flickr8k.token.txt           (captions)

This script:
- Loads all captions
- Matches them to image files
- Stores up to MAX_ITEMS entries in corpus.json
"""

import os
import json
from collections import defaultdict

# ====== FIXED PATHS FOR YOUR EXACT FOLDER STRUCTURE ======
IMAGES_ROOT = "Flickr8k_Dataset/Flicker8k_Dataset"
CAPTIONS_FILE = "Flickr8k_text/Flickr8k.token.txt"
# ==========================================================

MAX_ITEMS = 2000  # use 2000 images to keep later steps fast

OUT_DIR = "flickr_data"
OUT_CORPUS = os.path.join(OUT_DIR, "corpus.json")


def load_captions(path):
    """
    Loads captions from Flickr8k.token.txt
    """
    d = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # split at first tab or first space
            parts = line.split("\t")
            if len(parts) == 1:
                parts = line.split(" ", 1)
            if len(parts) < 2:
                continue

            img_tag = parts[0]  # e.g. "1000268201_693b08cb0e.jpg#0"
            caption = parts[1].strip()

            # remove #0/#1/#2...
            filename = img_tag.split("#")[0]
            d[filename].append(caption)
    return d


def prepare_corpus():
    print("Loading captions from:", CAPTIONS_FILE)
    captions_map = load_captions(CAPTIONS_FILE)
    print("Total caption entries:", len(captions_map))

    print("Loading images from:", IMAGES_ROOT)
    if not os.path.isdir(IMAGES_ROOT):
        raise FileNotFoundError(f"Image directory not found: {IMAGES_ROOT}")

    all_images = sorted([f for f in os.listdir(IMAGES_ROOT) if f.lower().endswith(".jpg")])
    print("Total images found:", len(all_images))

    items = []
    for fn in all_images:
        if fn in captions_map:
            img_path = os.path.join(IMAGES_ROOT, fn).replace("\\", "/")
            items.append({
                "image_path": img_path,
                "captions": captions_map[fn]
            })
        if len(items) >= MAX_ITEMS:
            break

    print(f"Using {len(items)} items for corpus (MAX_ITEMS={MAX_ITEMS}).")

    os.makedirs(OUT_DIR, exist_ok=True)

    corpus = []
    for i, item in enumerate(items):
        corpus.append({
            "id": i,
            "image_path": item["image_path"],
            "captions": item["captions"]
        })

    with open(OUT_CORPUS, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    print("Corpus saved at:", OUT_CORPUS)
    print("Example entry:")
    print(json.dumps(corpus[0], indent=2))


if __name__ == "__main__":
    prepare_corpus()
