# Multimodal RAG System: Story ↔ Image Understanding

## TEAM MEMBERS

- **202418041 : Palak Jain**
- **202418051 : Sheetal Jain**

---

## Project Overview

This project implements a **Multimodal Retrieval-Augmented Generation (RAG) system** that connects **textual stories** and **visual images** in a unified and explainable manner. The system demonstrates how pretrained multimodal and language models can be combined to perform reasoning across different data modalities without training any model from scratch.

The project focuses on two core capabilities:

1. **Story / Text → Image Retrieval**  
   Given a narrative paragraph or story written in natural language, the system retrieves the most semantically relevant image from a dataset by understanding the *meaning* of the story rather than relying on keyword matching.

2. **Image → Story Generation**  
   Given an image, the system generates a **multi-sentence narrative story** that is grounded in the visual content of the image using a Retrieval-Augmented Generation (RAG) framework.

The primary objective of this project is **deep conceptual understanding**—to clearly demonstrate how embedding, retrieval, and generation work together in a multimodal system that is CPU-friendly, modular, and academically explainable.

---

## Motivation

Humans naturally associate visual perception with language. When we see an image, we instinctively describe it using stories, emotions, and context. However, many traditional AI systems treat text and images as separate problems, limiting their ability to reason across modalities.

The motivation behind this project is to build a system that:
- Understands **long-form narrative text**, not just keywords
- Retrieves images based on **semantic meaning**
- Generates **grounded stories** instead of hallucinated descriptions
- Clearly separates **retrieval (understanding)** from **generation (expression)**

This project demonstrates how modern transformer-based models can be composed into a pipeline that mirrors human-like multimodal reasoning.

---

## Dataset

### Flickr8k Dataset

**Dataset Link:**  
👉 https://www.kaggle.com/datasets/adityajn105/flickr8k

The **Flickr8k dataset** is a standard benchmark dataset used in multimodal research. It contains:
- Approximately **8,000 real-world images**
- **Five human-written textual descriptions per image**

These descriptions are natural language captions that vary in wording and focus, making the dataset suitable for semantic understanding rather than rigid label-based learning.

### Why Flickr8k?

- Small enough to run on **CPU-only systems**
- High-quality human annotations
- Widely used in image–text alignment research
- Ideal for demonstrating multimodal retrieval and grounding

### Dataset Usage in This Project

- **Images** are used as retrieval candidates
- **Captions** act as grounding context for story generation

---

## System Architecture

The system follows a **modular three-layer architecture**, where each layer has a clearly defined responsibility. This design makes the system easy to understand, debug, and extend.

### 1. Embedding Layer (CLIP)

- Uses **CLIP (Contrastive Language–Image Pretraining)** by OpenAI
- Converts both images and text/stories into **512-dimensional embeddings**
- Maps both modalities into a shared semantic space

Because both images and text are embedded into the same vector space, they can be compared directly using similarity metrics.

---

### 2. Retrieval Layer

- Uses **cosine similarity** to measure semantic alignment between embeddings
- Fully semantic (not keyword-based)
- Supports:
  - **Story → Image retrieval**
  - **Image → Caption retrieval** (for grounding generation)

Retrieval ensures that generation is **grounded in real dataset information**, reducing hallucination.

---

### 3. Generation Layer (Retrieval-Augmented Generation)

- Uses **FLAN-T5**, an instruction-tuned language model
- Follows the **Retrieval-Augmented Generation (RAG)** paradigm
- Retrieved captions are injected into a structured prompt
- Prompt engineering enforces:
  - Narrative structure
  - Minimum story length
  - Multi-sentence output

Separating retrieval and generation allows story quality to be improved through prompt design rather than retraining large models.

---

## Workflow

### A. Story / Text → Image Retrieval

1. User enters a narrative story through the Streamlit UI.
2. The story is tokenized and truncated to CLIP’s maximum limit (77 tokens).
3. CLIP’s text encoder converts the story into a **512-dimensional embedding**.
4. This embedding is compared against **precomputed image embeddings**.
5. Cosine similarity is used to find the most semantically aligned image.
6. The retrieved image is displayed in the UI.

This workflow enables image retrieval based on **meaning rather than keywords**.

---

### B. Image → Story Generation

1. User provides an image path via the UI.
2. The image is encoded using CLIP’s image encoder.
3. Relevant captions are retrieved using cosine similarity.
4. Captions are inserted into a structured narrative prompt.
5. FLAN-T5 generates a detailed story (length controlled via slider).
6. The generated story is displayed alongside the image.

This workflow follows a **Retrieval-Augmented Generation (RAG)** approach, ensuring factual grounding.

---

## Project Files and Their Roles

- **prepare_flickr.py**  
  Reads Flickr8k captions, associates them with image paths, and creates a structured corpus.

- **build_index.py**  
  Precomputes CLIP image and text embeddings and stores them on disk for efficient retrieval.

- **rag_retriever.py**  
  Implements semantic retrieval using cosine similarity for both workflows.

- **rag_generator.py**  
  Implements RAG-based story generation using FLAN-T5 and prompt engineering.

- **app.py**  
  Streamlit-based user interface that connects all components and enables interactive testing.

---

## User Interface

The project includes a simple **Streamlit UI** with two modes:

### Story → Image
User inputs a narrative story and the system retrieves the most relevant image.

### Image → Story
User provides an image path and controls story length using a slider. The system generates a grounded narrative story.

---

## Results

### Story → Image Retrieval Output

![Input](https://github.com/SheetalJain0505/T12_DS635_MLSE_Project/blob/main/Stroy_to_Image.png)
![Output](https://github.com/SheetalJain0505/T12_DS635_MLSE_Project/blob/main/Output_Image_generator.png)

### Image → Story Generation Output

![Input](https://github.com/SheetalJain0505/T12_DS635_MLSE_Project/blob/main/Image_to_Story.png)
![Output](https://github.com/SheetalJain0505/T12_DS635_MLSE_Project/blob/main/Output_Story_generator.png)

The results demonstrate that the system:
- Retrieves images based on semantic meaning
- Generates coherent and grounded stories
- Avoids hallucination through retrieval grounding

---

## Limitations

- Uses FLAN-T5-base, which limits creativity compared to larger models
- Story length is bounded by CPU constraints
- CLIP text truncation may remove some contextual details

These limitations were intentionally accepted to keep the system lightweight and explainable.

---

## Conclusion

This project demonstrates a complete **multimodal RAG pipeline** that integrates embedding, retrieval, and generation into a unified system. By focusing on **Story → Image** and **Image → Story** tasks, the project highlights how retrieval enhances generation while maintaining factual grounding.

The system emphasizes **understanding over complexity**, making it well-suited for academic learning, demonstrations, and viva discussions.

---

## How to Run

```bash
git clone <repository-url>
cd simple_rag_multimodal

python -m venv rag_env
rag_env\Scripts\activate

pip install torch transformers sentence-transformers pillow streamlit numpy tqdm

python prepare_flickr.py
python build_index.py

streamlit run app.py
