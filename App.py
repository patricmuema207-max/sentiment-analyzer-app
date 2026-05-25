import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🤖"
)

st.title("🤖 AI Customer Review Sentiment Analyzer")

@st.cache_resource
def load_model():

    classifier = pipeline(
        "text-classification",
        model="patricmuema20/patric-sentiment-model"
    )

    return classifier

classifier = load_model()

review = st.text_area(
    "Enter customer review"
)

if st.button("Analyze"):

    if review.strip() != "":

        result = classifier(review)

        label = result[0]["label"]
        score = result[0]["score"]

        st.subheader("Prediction")

        st.write(f"Label: {label}")

        st.write(f"Confidence: {score:.2f}")

    else:

        st.warning("Please enter a review.")
