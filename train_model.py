import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib

# ---- NLTK stopwords ----
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

# ---- Charger la base ----
df = pd.read_csv("spam (or) ham.csv", encoding="latin-1")

# تنظيف أسماء الأعمدة لتجنب المشاكل
df.columns = df.columns.str.strip().str.lower()  # يحيد المسافات ويحول الأحرف لصغيرة

# التحقق من الأعمدة المطلوبة
if 'class' not in df.columns or 'sms' not in df.columns:
    raise ValueError("⚠️ Le CSV doit contenir les colonnes 'class' et 'sms' (case insensitive).")

df = df[['class', 'sms']]

# Nettoyage basique
df['class'] = df['class'].str.strip().str.lower()
df['sms'] = df['sms'].astype(str)

# Garder uniquement ham/spam
df = df[df['class'].isin(['ham', 'spam'])].dropna().reset_index(drop=True)

# ---- Nettoyage texte ----
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

df["cleaned"] = df["sms"].apply(clean_text)
df = df[df["cleaned"].str.strip() != ""].reset_index(drop=True)

# ---- Préparation ----
X = df["cleaned"]
y = df["class"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- TF-IDF fiable ----
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# ---- Modèle fiable ----
model = LogisticRegression(
    max_iter=1500,
    solver="liblinear",
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ---- Évaluation ----
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# ---- Sauvegarde ----
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf.pkl")

print("Modèle sauvegardé avec succès!")
