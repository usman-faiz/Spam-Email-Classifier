import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit app interface
st.title("Spam Email Classifier")
st.subheader("Paste your email content below:")

# Input field
email_input = st.text_area("Email Content", height=200)

if st.button("Classify"):
    if email_input.strip():
        # Preprocess and classify
        email_vector = vectorizer.transform([email_input])
        prediction = model.predict(email_vector)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        st.write(f"**Classification Result:** {result}")
    else:
        st.warning("Please enter email content!")
