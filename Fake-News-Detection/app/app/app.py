import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Page title
st.title("📰 Fake News Detection System")
st.write("Enter a news article and click Predict.")

# Text input
news = st.text_area("Enter News Article")

# Prediction button
if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:
        # Convert to TF-IDF
        news_vector = vectorizer.transform([news])

        # Predict
        prediction = model.predict(news_vector)

        # Confidence score
        confidence = model.predict_proba(news_vector)

        if prediction[0] == 0:
            st.error("❌ FAKE NEWS")
            st.write(
                f"Confidence: {max(confidence[0])*100:.2f}%"
            )
        else:
            st.success("✅ REAL NEWS")
            st.write(
                f"Confidence: {max(confidence[0])*100:.2f}%"
            )