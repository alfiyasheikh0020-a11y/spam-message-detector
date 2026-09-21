import streamlit as st
import joblib
import string

# Load trained model and TF-IDF vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


def predict_message(message):
    if not message.strip():
        return "⚠️ Please enter a message."

    message = message.lower()

    message = message.translate(
        str.maketrans("", "", string.punctuation)
    )

    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)[0]

    if prediction == 1:
        return "🚨 SPAM MESSAGE"
    else:
        return "✅ NOT SPAM"


# Page settings
st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📱",
    layout="wide"
)

# Title
st.markdown(
    "<h1 style='text-align:center;'>📱 Spam Message Detector</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Machine Learning & NLP Based Spam Detection System</p>",
    unsafe_allow_html=True
)

st.divider()

# Input
st.subheader("🔍 Check Your Message")

message = st.text_area(
    "💬 Enter or paste your message:",
    placeholder="Type or paste your message here...",
    height=180
)

# Button
if st.button("🔍 Check Message", type="primary"):

    result = predict_message(message)

    st.subheader("📊 Prediction")

    if "SPAM" in result:
        st.error(result)
    elif "NOT SPAM" in result:
        st.success(result)
    else:
        st.warning(result)

st.divider()

# Project information
st.markdown(
    """
    **Model:** Multinomial Naive Bayes  
    **Text Representation:** TF-IDF  
    **Dataset:** SMS Spam Collection  
    **Accuracy:** 95.70%
    """
)
