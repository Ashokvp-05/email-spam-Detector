import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📩 SMS Spam Classifier")
st.write("Enter any message below to check if it is Spam or Not Spam.")

message = st.text_area("Message:")

if st.button("Check"):
    if message.strip() == "":
        st.warning("Please enter a valid message.")
    else:
        msg_vec = vectorizer.transform([message])
        result = model.predict(msg_vec)[0]

        if result == "spam":
            st.error("🚨 SPAM Message Detected!")
        else:
            st.success("✅ Not Spam")
