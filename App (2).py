import streamlit as st
from transformers import pipeline

st.title("AI Customer Review Sentiment Analyzer")
st.write("Enter a customer review below to analyze if it's **POSITIVE** or **NEGATIVE**")

# Your actual model from Hugging Face
@st.cache_resource  # This loads the model only once (saves memory)
def load_model():
    return pipeline(
        "text-classification",
        model="patricmuema20/patric-sentiment-model"
    )

classifier = load_model()

review = st.text_area("Enter customer review", height=150)

if st.button("Analyze Sentiment"):
    if review.strip() != "":
        with st.spinner("Analyzing..."):
            result = classifier(review)
        
        st.subheader("📊 Prediction Result")
        
        # Clean up the output
        sentiment = result[0]['label']
        confidence = result[0]['score']
        
        if sentiment == "POSITIVE" or sentiment == "LABEL_1":
            st.success(f"✅ **Sentiment: POSITIVE**")
        else:
            st.error(f"❌ **Sentiment: NEGATIVE**")
        
        st.write(f"Confidence: **{confidence:.2%}**")
        
        # Show raw result too
        with st.expander("View raw output"):
            st.write(result)
    else:
        st.warning("⚠️ Please enter a review before analyzing.")