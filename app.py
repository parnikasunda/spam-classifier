import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load saved model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return " ".join(words)


# ===================== UI DESIGN ===================== #

st.set_page_config(page_title="Spam Classifier", layout="centered")

# Custom CSS (for dark theme look)
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

# Input box
input_sms = st.text_area("Enter the Message")

# Dropdown (like your UI)
option = st.selectbox(
    "You Got Message From :-",
    ["Via Email", "Via SMS"]
)

# Checkbox
check = st.checkbox("Check me")

# Button
if st.button("Click to Predict"):

    if input_sms.strip() == "":
        st.warning("Please enter a message")
    else:
        # 1. preprocess
        transformed_sms = transform_text(input_sms)

        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. predict
        result = model.predict(vector_input)[0]

        # 4. display
        if result == 1:
            st.error("Spam 🚨")
        else:
            st.success("Not Spam ✅")