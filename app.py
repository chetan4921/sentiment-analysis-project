import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from langdetect import detect

# Load multilingual sentiment model
@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentiment = load_model()

# UI
st.title("🌍 Multilingual Sentiment Analysis")
st.write("Analyze sentiment of text in **100+ languages** using XLM-RoBERTa")

user_input = st.text_area("✍️ Enter text in any language:")

if st.button("Analyze"):
    if user_input.strip():
        try:
            lang = detect(user_input)
            result = sentiment(user_input)[0]
            
            # Label mapping
            label_map = {
                "LABEL_0": "Negative",
                "LABEL_1": "Neutral",
                "LABEL_2": "Positive"
            }
            sentiment_label = label_map.get(result["label"], result["label"])
            
            # Display results
            st.write(f"**Language Detected:** {lang.upper()}")
            st.write(f"**Sentiment:** {sentiment_label}")
            st.write(f"**Confidence Score:** {result['score']:.2f}")
            
            # Emoji feedback
            if sentiment_label == "Positive":
                st.success("😊 Positive sentiment detected!")
            elif sentiment_label == "Negative":
                st.error("😡 Negative sentiment detected!")
            else:
                st.info("😐 Neutral sentiment detected.")
        
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter some text.")
