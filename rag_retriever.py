import numpy as np
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
import json

CORPUS_PATH = "flickr_data/corpus.json"
IMG_EMB_PATH = "flickr_data/image_embeddings.npy"

class RAGRetriever:

    def __init__(self):
        print("Loading embeddings...")
        self.image_embeds = np.load(IMG_EMB_PATH)

        print("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        print("Loading text encoder...")
        self.text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        print("Loading corpus metadata...")
        with open(CORPUS_PATH, "r") as f:
            self.corpus = json.load(f)

    def cosine_sim(self, a, b):
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(b, a)

    # ------------------------------------
    # TEXT → IMAGE
    # ------------------------------------
    def text_to_image(self, story_text, top_k=3):
        """
        Takes a full story / paragraph as input and retrieves the most relevant image.
        """

        print("\nSTORY QUERY:")
        print(story_text)

        # Truncate story to CLIP's max token limit
        tokens = self.clip_processor.tokenizer(
            story_text,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        with torch.no_grad():
            story_embedding = self.clip_model.get_text_features(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"]
            ).cpu().numpy()[0]

        scores = self.cosine_sim(story_embedding, self.image_embeds)
        idx = scores.argsort()[-top_k:][::-1]

        return [(self.corpus[i]["image_path"], float(scores[i])) for i in idx]


    # ------------------------------------
    # IMAGE → STORY (CLIP text encoder)
    # ------------------------------------
    def image_to_story(self, image_path, top_k=3):
        img = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            q_emb = self.clip_model.get_image_features(**inputs).cpu().numpy()[0]

        clip_text_embeds = []
        for item in self.corpus:
            text = " ".join(item["captions"])

            tokens = self.clip_processor.tokenizer(
                text,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            )

            with torch.no_grad():
                emb = self.clip_model.get_text_features(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"]
                ).cpu().numpy()[0]

            clip_text_embeds.append(emb)

        clip_text_embeds = np.array(clip_text_embeds)

        scores = self.cosine_sim(q_emb, clip_text_embeds)
        idx = scores.argsort()[-top_k:][::-1]
        return [(self.corpus[i]["captions"], float(scores[i])) for i in idx]
