from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rag_retriever import RAGRetriever
from PIL import Image

class RAGStoryGenerator:

    def __init__(self):
        print("Loading generator model...")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

        print("Loading retriever...")
        self.retriever = RAGRetriever()

    # ----------------------------------------------------
    # IMAGE → STORY GENERATION (this is the missing method)
    # ----------------------------------------------------
    def generate_story_from_image(self, image_path, story_length=180):
        """
        Generates a multi-paragraph story from an image using RAG.
        """

        # 1. Retrieve captions
        retrieved = self.retriever.image_to_story(image_path, top_k=3)
        captions = " ".join([" ".join(item[0]) for item in retrieved])

        # 2. Strong narrative prompt with length control
        prompt = f"""
    You are a creative children's story writer.

    IMPORTANT RULES:
    - Write a detailed story of AT LEAST 100 WORDS.
    - The story must have multiple sentences and feel like a real narrative.
    - Describe actions, emotions, surroundings, and progression over time.
    - Do NOT summarize.
    - Do NOT write a caption.
    - Write like a short story from a book.

    Example:
    Image descriptions:
    A boy is playing with a dog in the park.

    Story:
    The boy ran happily across the green park while his dog chased him with excitement.
    They laughed and played together for hours, enjoying the warmth of the afternoon sun.
    As they explored the open space, the boy felt free and joyful, forgetting everything else.

    When the day slowly came to an end, the boy sat on the grass beside his dog, feeling tired
    but content. The park grew quiet as the sky changed colors, and the peaceful moment stayed
    with him long after they went home.

    Now write a NEW story of AT LEAST 100 WORDS.

    Image descriptions:
    {captions}

    Story:
    """

        tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        # 3. Force longer generation
        output = self.model.generate(
            tokens["input_ids"],
            max_length=story_length,
            min_length=int(story_length * 0.6),  # 🔥 VERY IMPORTANT
            do_sample=True,
            temperature=0.95,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)





    # ----------------------------------------------------
    # TEXT → STORY + RETRIEVED IMAGES
    # ----------------------------------------------------
    def generate_story_from_text(self, query):
        retrieved = self.retriever.text_to_image(query, top_k=3)
        paths = [x[0] for x in retrieved]

        prompt = f"Write a descriptive story about: {query}"

        tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        output = self.model.generate(tokens["input_ids"], max_length=120)

        story = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return paths, story
