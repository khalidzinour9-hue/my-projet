import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle
import re
import nltk
from nltk.corpus import stopwords

# If running first time, uncomment the following:
# nltk.download('stopwords')
nltk.download('punkt')
nltk.download('stopwords')
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

def main():
    df = pd.read_csv("spam.csv", encoding="latin-1")[["v1","v2"]]
    df.columns = ["label","text"]
    df["label"] = df["label"].map({"ham":0, "spam":1})
    df["clean_text"] = df["text"].apply(clean_text)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 macro:", f1_score(y_test, y_pred, average="macro"))
    print(classification_report(y_test, y_pred))

    # Save artifacts
    with open("spam_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("tfidf.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("Model and vectorizer saved as spam_model.pkl and tfidf.pkl")

if __name__ == "__main__":
    main()
