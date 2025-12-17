import streamlit as st
from PIL import Image
from rag_retriever import RAGRetriever
from rag_generator import RAGStoryGenerator

st.set_page_config(
    page_title="Multimodal RAG Demo",
    layout="centered"
)

st.title("🧠 Multimodal RAG System")
st.write("Story → Image retrieval and Image → Story generation")

# ---------------------------------
# Load models only once
# ---------------------------------
@st.cache_resource
def load_models():
    retriever = RAGRetriever()
    generator = RAGStoryGenerator()
    return retriever, generator

R, G = load_models()

# ---------------------------------
# Sidebar
# ---------------------------------
mode = st.sidebar.radio(
    "Choose Functionality",
    [
        "📖 Story → Image",
        "🖼️ Image → Story"
    ]
)

# ---------------------------------
# STORY → IMAGE
# ---------------------------------
if mode == "📖 Story → Image":
    st.header("📖 Story → Image Retrieval")

    story_input = st.text_area(
        "Enter a story or paragraph:",
        height=180,
        placeholder=(
            "A young girl was laughing happily as she jumped from one couch "
            "to another inside her living room, enjoying her playful afternoon."
        )
    )

    if st.button("Find Image"):
        if story_input.strip() == "":
            st.warning("Please enter a story.")
        else:
            results = R.text_to_image(story_input)
            img_path = results[0][0]

            st.success("Most relevant image retrieved:")
            image = Image.open(img_path)
            st.image(image, caption=img_path, use_column_width=True)

# ---------------------------------
# IMAGE → STORY
# ---------------------------------
if mode == "🖼️ Image → Story":
    st.header("🖼️ Image → Story Generation")

    image_path = st.text_input(
        "Enter image path:",
        placeholder="Flickr8k_Dataset/Flicker8k_Dataset/667626_18933d713e.jpg"
    )

    story_length = st.slider(
        "Max Story length",
        min_value=80,
        max_value=300,
        value=150,
        step=20
    )

    if st.button("Generate Story"):
        if image_path.strip() == "":
            st.warning("Please enter an image path.")
        else:
            try:
                image = Image.open(image_path)
                st.image(image, caption="Input Image", use_column_width=True)

                story = G.generate_story_from_image(
                    image_path,
                    story_length=story_length
                )

                st.subheader("📖 Generated Story")
                st.write(story)

            except Exception as e:
                st.error(f"Error: {e}")
