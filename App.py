import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🤖")
st.title("🤖 AI Customer Review Sentiment Analyzer")
st.markdown("---")

# Load YOUR custom model from Hugging Face
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="patricmuema20/patric-sentiment-model"
    )

try:
    classifier = load_model()
    st.success("✅ Custom model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

st.subheader("📝 Enter Customer Review")
review = st.text_area("", height=150, placeholder="Type or paste a customer review here...")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True)

if analyze_button:
    if review.strip() == "":
        st.warning("⚠️ Please enter a review before analyzing.")
    else:
        with st.spinner("Analyzing with my custom-trained model..."):
            try:
                result = classifier(review)
                
                sentiment = result[0]['label']
                confidence = result[0]['score']
                
                st.markdown("---")
                st.subheader("📊 Analysis Result")
                
                # Handle different label formats
                if sentiment in ["POSITIVE", "LABEL_1", "1", 1]:
                    st.success(f"### ✅ POSITIVE SENTIMENT")
                    st.balloons()
                else:
                    st.error(f"### ❌ NEGATIVE SENTIMENT")
                
                st.metric("Confidence Score", f"{confidence:.2%}")
                
                with st.expander("🔬 View Detailed Output"):
                    st.write(f"**Label:** {sentiment}")
                    st.write(f"**Confidence:** {confidence:.4f}")
                    st.write(f"**Raw Result:** {result}")
                    
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

st.markdown("---")
st.caption("🚀 Powered by Hugging Face Transformers | Fine-tuned DistilBERT Model")
st.caption(f"📦 Model: `patricmuema20/patric-sentiment-model`")
