import streamlit as st
import pickle
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

# Download NLTK resources (run only once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load pre-trained models
with open("multinomial_nb_model.pkl", "rb") as f:
    nb_model = pickle.load(f)
with open("random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:  # Load the TF-IDF vectorizer
    vectorizer = pickle.load(f)



# Define text preprocessing functions
def remove_html(text):
    html_pattern = re.compile(r'<.*?>')
    return html_pattern.sub(r'', text)

def remove_url(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

def apply_stemming(text):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in text.split()])

def apply_lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def preprocess_text(text):
    text = remove_html(text)
    text = remove_url(text)
    text = remove_emojis(text)
    text = remove_stopwords(text)
    text = apply_stemming(text)
    text = apply_lemmatization(text)
    return text

# Streamlit app
st.title("Text Classification App")

input_text = st.text_area("Enter the text to classify:")

if st.button("Classify"):
    if input_text:

        # Preprocess the input text
        preprocessed_text = preprocess_text(input_text)

        # Transform the input text using the loaded TF-IDF vectorizer
        text_vector = vectorizer.transform([preprocessed_text])
        
        # Make predictions using the loaded models
        nb_pred = nb_model.predict(text_vector)
        rf_pred = rf_model.predict(text_vector)
        
        # Display predictions
        if nb_pred[0] == 0:
            prediction= 'Fake'
        else:
            prediction = 'Real'
        st.subheader(f"MultinomialNB Prediction: {prediction}")
        if rf_pred[0] == 0:
            rf_prediction= 'Fake'
        else:
            rf_prediction = 'Real'
        st.subheader(f"RandomForest Prediction: {rf_prediction}")
    else:
        st.error("Please enter some text for classification.")
