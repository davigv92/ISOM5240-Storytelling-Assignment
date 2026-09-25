import streamlit as st
from transformers import pipeline
from gtts import gTTS
from PIL import Image


# Image to text
def img2text(image):
    image_to_text_model = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )
    text = image_to_text_model(image)[0]["generated_text"]
    return text


# Text to story
story_generator = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-360M-Instruct"
)


def text2story(text):
    messages = [
        {
            "role": "user",
            "content": (
                f"Write a fun and simple story for children aged 3 to 10 "
                f"based on this image description: {text}. "
                f"The story should be between 50 and 100 words. "
                f"Return only the story."
            )
        }
    ]

    result = story_generator(
        messages,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.2
    )

    story_text = result[0]["generated_text"][-1]["content"]

    return story_text.strip()


# Text to audio
def text2audio(story_text):
    tts = gTTS(text=story_text, lang="en")
    audio_file = "story.mp3"
    tts.save(audio_file)
    return audio_file


# Streamlit application
st.title("📖 AI Storytelling Adventure")
st.write("Upload an image and create a fun story for children aged 3–10.")


uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_image is not None:

    image = Image.open(uploaded_image)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("✨ Create Story"):

        # 1. Image → caption
        with st.spinner("Looking at the image..."):
            caption = img2text(image)

        st.write("**Image description:**", caption)

        # 2. Caption → story
        with st.spinner("Writing the story..."):
            story = text2story(caption)

        st.subheader("📖 Your Story")
        st.write(story)

        st.write("Word count:", len(story.split()))

        # 3. Story → audio
        with st.spinner("Creating the audio..."):
            audio_file = text2audio(story)

        st.subheader("🔊 Listen to the story")
        st.audio(audio_file, format="audio/mp3")
