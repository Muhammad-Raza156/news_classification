import numpy as np
import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')


# Data cleaning functions
def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', str(text))

def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', str(text))

def remove_emojis(text):
    emoji_pattern = re.compile("[" +
        u"\U0001F600-\U0001F64F" +  # emoticons
        u"\U0001F300-\U0001F5FF" +  # symbols & pictographs
        u"\U0001F680-\U0001F6FF" +  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF" +  # flags
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def clean_text(text):
    text = remove_url(text)
    text = remove_html(text)
    text = remove_emojis(text)
    return text

# Read and preprocess data
def read_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['title_without_stopwords', 'text_without_stopwords', 'label'])
    df['label'] = LabelEncoder().fit_transform(df['label'])
    df['title_without_stopwords'] = df['title_without_stopwords'].apply(clean_text)
    df['text_without_stopwords'] = df['text_without_stopwords'].apply(clean_text)
    df['text'] = df['text_without_stopwords'] + " " + df['title_without_stopwords']

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    df['text'] = df['text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
    
    df['text'] = df['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

    return df[['text', 'label']]

def format_dataset(data):
    X = data['text']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    return X_train_tfidf.toarray(), y_train, X_test_tfidf.toarray(), y_test, tfidf

def gaussianNB_classifier(X_train, y_train, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred)

def multinomialNB_classifier(X_train, y_train, X_test, y_test):
    mnb_model = MultinomialNB()
    mnb_model.fit(X_train, y_train)
    y_pred = mnb_model.predict(X_test)
    with open("multinomial_nb_model.pkl", "wb") as f:
        pickle.dump(mnb_model, f)
    return mnb_model, accuracy_score(y_test, y_pred), classification_report(y_test, y_pred)

def rf_classifier(X_train, y_train, X_test, y_test):
    rf_model = RandomForestClassifier(n_estimators=140, max_depth=45, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    with open("random_forest_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    return rf_model, roc_auc_score(y_test, y_pred_proba), accuracy_score(y_test, y_pred), classification_report(y_test, y_pred), y_pred_proba

def roc_plot(y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label="AUC-ROC")
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()

# Run the script
if __name__ == "__main__":
    file_path= r"C:\Users\AIMS TECH\Downloads\text_classification_implementation\news_articles.csv"
    data = read_data(file_path)
    X_train, y_train, X_test, y_test, tfidf = format_dataset(data)
    
    # Train and evaluate GaussianNB
    gnb_accuracy, gnb_report = gaussianNB_classifier(X_train, y_train, X_test, y_test)
    print(f"GaussianNB Accuracy: {gnb_accuracy:.4f}")
    print("Classification Report for GaussianNB:\n", gnb_report)

    # Train and evaluate MultinomialNB
    mnb_model,mnb_accuracy, mnb_report= multinomialNB_classifier(X_train, y_train, X_test, y_test)
    print(f"MultinomialNB Accuracy: {mnb_accuracy:.4f}")
    print("Classification Report for MultinomialNB:\n", mnb_report)

    # Train and evaluate RandomForest
    rf_model, rf_auc, rf_accuracy, rf_report, rf_y_pred_proba= rf_classifier(X_train, y_train, X_test, y_test)
    print(f"RandomForest AUC-ROC Score: {rf_auc:.4f}")
    print(f"RandomForest Accuracy: {rf_accuracy:.4f}")
    print("Classification Report for RandomForest:\n", rf_report)

    # Plot ROC curve for RandomForest
    roc_plot(y_test, rf_y_pred_proba)

    # Save models
    # with open("multinomial_nb_model.pkl", "wb") as f:
    #     pickle.dump(mnb_model, f)
    # with open("random_forest_model.pkl", "wb") as f:
    #     pickle.dump(rf_model, f)
    with open("count_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
