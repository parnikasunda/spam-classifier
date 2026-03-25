# version 2
import streamlit as st
import pickle
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # ✅ FIX

# Load saved model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Initialize stemmer
ps = PorterStemmer()

# ✅ FIX: use sklearn stopwords
stop_words = ENGLISH_STOP_WORDS

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)


# ===================== UI DESIGN ===================== #

st.set_page_config(page_title="Spam Classifier", layout="centered")

st.markdown("""
<style>
body {
    background-color: #0b0f19;
}
.main {
    background-color: #0b0f19;
}
h1 {
    color: #ffffff;
    font-size: 48px;
}
.stTextInput>div>div>input {
    background-color: #2a2f3a;
    color: white;
}
.stButton>button {
    background-color: #1f77ff;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# ===================== HEADER ===================== #

st.title("Email Spam Classifier")

input_sms = st.text_area("Enter the Message")

option = st.selectbox(
    "You Got Message From :-",
    ["Via Email", "Via SMS"]
)

check = st.checkbox("Check me")

if st.button("Click to Predict"):

    if input_sms.strip() == "":
        st.warning("Please enter a message")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("Spam 🚨")
        else:
            st.success("Not Spam ✅")
