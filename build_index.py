import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
import os

CORPUS_PATH = "flickr_data/corpus.json"
OUT_DIR = "flickr_data"

def load_image(path):
    return Image.open(path).convert("RGB")

def main():
    print("Loading corpus...")
    with open(CORPUS_PATH, "r") as f:
        corpus = json.load(f)

    print(f"Corpus size: {len(corpus)}")

    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    torch_dtype="auto",
    use_safetensors=True
    )


    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("Loading SentenceTransformer...")
    text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    image_embeds = []
    text_embeds = []

    print("Computing image embeddings...")
    for item in tqdm(corpus):
        img = load_image(item["image_path"])
        inputs = clip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs).cpu().numpy()
        image_embeds.append(emb[0])

    print("Computing text embeddings...")
    for item in tqdm(corpus):
        all_caps = " ".join(item["captions"])
        emb = text_model.encode(all_caps)
        text_embeds.append(emb)

    image_embeds = np.array(image_embeds)
    text_embeds = np.array(text_embeds)

    os.makedirs(OUT_DIR, exist_ok=True)

    np.save(f"{OUT_DIR}/image_embeddings.npy", image_embeds)
    np.save(f"{OUT_DIR}/text_embeddings.npy", text_embeds)

    print("\nSaved files:")
    print(f"- {OUT_DIR}/image_embeddings.npy")
    print(f"- {OUT_DIR}/text_embeddings.npy")

if __name__ == "__main__":
    main()
