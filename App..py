import streamlit as st
from transformers import pipeline

st.title("AI Customer Review Sentiment Analyzer")

classifier = pipeline(
    "text-classification",
    model="patricmuema20/patric-sentiment-model"
)

review = st.text_area("Enter customer review")

if st.button("Analyze"):

    if review.strip() != "":

        result = classifier(review)

        st.subheader("Prediction Result")

        st.write(result)

    else:

        st.warning("Please enter a review.")
