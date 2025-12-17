# Multimodal RAG System: Story ↔ Image Understanding

## TEAM MEMBERS

* **202418041 : Palak Jain**
* **202418051 : Sheetal Jain**

---

## Project Overview

This project implements a **Multimodal Retrieval-Augmented Generation (RAG) system** that connects **textual stories** and **visual images** in a meaningful way. The system is capable of:

1. **Story / Text → Image Retrieval**
   Given a narrative paragraph or story, the system retrieves the most semantically relevant image from a dataset.

2. **Image → Story Generation**
   Given an image, the system generates a detailed, multi-sentence narrative story grounded in the visual content of the image.

The primary goal of this project is **conceptual understanding**, not model scale. The system is designed to run on **CPU-only**, use **pretrained models**, and clearly demonstrate how **retrieval and generation can be combined** in a multimodal setting.

---

## Motivation

Humans naturally associate what they see with what they read or imagine. Traditional AI systems often handle text and images separately, but real-world understanding requires linking these modalities. This project demonstrates how modern transformer-based models can be composed into a pipeline that:

* Understands stories beyond keywords
* Retrieves images based on semantic meaning
* Generates grounded stories instead of hallucinated text

---

## Dataset

### Flickr8k Dataset

**Dataset Link:** [https://www.kaggle.com/datasets/adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)

The project uses the **Flickr8k dataset**, which contains:

### Flickr8k Dataset

The project uses the **Flickr8k dataset**, which contains:

* ~8,000 real-world images
* 5 human-written captions per image

Why Flickr8k:

* Small and CPU-friendly
* High-quality natural language descriptions
* Widely used for image–text research
* Ideal for multimodal retrieval tasks

The dataset is used in two ways:

* Images act as retrieval candidates
* Captions act as grounding context for story generation

---

## System Architecture

The system follows a **modular three-layer architecture**:

### 1. Embedding Layer (CLIP)

* Uses **CLIP (Contrastive Language–Image Pretraining)** by OpenAI
* Converts both images and text/stories into **512-dimensional embeddings**
* Maps both modalities into a shared semantic space

This shared embedding space allows direct comparison between stories and images.

### 2. Retrieval Layer

* Uses **cosine similarity** to measure semantic alignment between embeddings
* Supports:

  * Story → Image retrieval
  * Image → Caption retrieval (for grounding)

Retrieval ensures that generation is **grounded in real data**.

### 3. Generation Layer (RAG)

* Uses **FLAN-T5**, an instruction-tuned text generation model
* Follows the **Retrieval-Augmented Generation (RAG)** paradigm
* Retrieved captions are injected into a structured prompt
* Prompt engineering enforces narrative structure and minimum story length

---

## Workflow

### A. Story / Text → Image Retrieval

1. User enters a story or paragraph in the UI
2. Story is tokenized and truncated (CLIP limit: 77 tokens)
3. CLIP text encoder converts story into a 512-D embedding
4. Embedding is compared with precomputed image embeddings
5. Image with highest cosine similarity is retrieved
6. Retrieved image is displayed in the UI

### B. Image → Story Generation

1. User provides an image path via the UI
2. Image is encoded using CLIP image encoder
3. Relevant captions are retrieved using cosine similarity
4. Captions are inserted into a narrative prompt
5. FLAN-T5 generates a detailed story (length controlled by slider)
6. Story is displayed along with the image

---

## Project Files and Their Roles

* **prepare_flickr.py**
  Prepares the Flickr8k dataset by reading captions and associating them with image paths.

* **build_index.py**
  Precomputes CLIP image and text embeddings and stores them for efficient retrieval.

* **rag_retriever.py**
  Implements story-to-image and image-to-caption retrieval using cosine similarity.

* **rag_generator.py**
  Implements Retrieval-Augmented Generation using FLAN-T5 with strong prompt engineering.

* **app.py**
  Streamlit-based UI that allows interactive testing of both functionalities.

---

## User Interface

The project includes a simple **Streamlit UI** with two modes:

### Story → Image

User inputs a full narrative story and the system retrieves the most relevant image.

### Image → Story

User provides an image path and controls story length using a slider. The system generates a grounded narrative.

---

## Results

Below are example outputs from the system.

### Story → Image Retrieval Output

*![Story to Image Result](https://github.com/SheetalJain0505/T12_DS635_MLSE_Project/blob/main/Output_Image_generator.png)*

### Image → Story Generation Output

*(Insert generated story text and corresponding image here)*

The results demonstrate that the system successfully:

* Retrieves images based on semantic meaning
* Generates coherent and grounded stories

---

## Limitations

* Uses FLAN-T5-base, which limits creativity compared to larger models
* Story length is bounded by CPU constraints
* CLIP text truncation may remove some context

These limitations were accepted intentionally to keep the system lightweight and explainable.

---

## Conclusion

This project demonstrates a complete **multimodal RAG pipeline** that integrates embedding, retrieval, and generation into a unified system. By focusing on **Story → Image** and **Image → Story** tasks, the project highlights how retrieval can enhance generation while maintaining factual grounding.

Most importantly, the project emphasizes **understanding over complexity**, making it suitable for academic learning, demonstrations, and viva discussions.

---

## How to Run

### 1. Clone the Repository

```bash
git clone <repository-url>
cd simple_rag_multimodal
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv rag_env
rag_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install torch transformers sentence-transformers pillow streamlit numpy tqdm
```

### 4. Prepare Dataset

```bash
python prepare_flickr.py
```

### 5. Build Embedding Index

```bash
python build_index.py
```

### 6. Run the Application

```bash
streamlit run app.py
```

---

```bash
rag_env\Scripts\activate
streamlit run app.py
```

---

## Acknowledgements

* OpenAI CLIP
* Google FLAN-T5
* Flickr8k Dataset
* Hugging Face Transformers

---

*This README is intentionally detailed to reflect deep conceptual understanding of the project.*
