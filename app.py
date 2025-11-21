import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Détecteur Spam or Ham", page_icon="📩")

# -----------------------------
# Inject CSS
# -----------------------------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------------
# NLTK stopwords
# -----------------------------
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# -----------------------------
# Charger modèle + vectorizer
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("spam_model.pkl")
        vectorizer = joblib.load("tfidf.pkl")
        return model, vectorizer
    except:
        return None, None

model, vectorizer = load_model()
if model is None or vectorizer is None:
    st.error("❌ Les fichiers du modèle sont introuvables. Lance d'abord train_model.py")
    st.stop()

# -----------------------------
# Nettoyage texte
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# -----------------------------
# UI principal
# -----------------------------
st.title("📩 Détecteur de Spam or Ham")

message = st.text_area("Écris ton message ici :")

if st.button("Analyser"):
    if message.strip() == "":
        st.warning("⚠️ Veuillez entrer un message.")
    else:
        cleaned = clean_text(message)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        label = "✔ Ham" if pred == 0 else "❌ SPAM"
        st.success(f"Résultat : **{label}**")

# -----------------------------
# Prédiction CSV
# -----------------------------
uploaded = st.file_uploader("Importer un fichier CSV (colonne: sms)", type=["csv"])

def read_csv_safely(file):
    encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
    for enc in encodings:
        try:
            return pd.read_csv(file, encoding=enc)
        except:
            continue
    raise ValueError("⚠️ Impossible de lire le fichier CSV — encodage non supporté.")

if uploaded:
    try:
        df = read_csv_safely(uploaded)
        if "sms" not in df.columns:
            st.error("⚠️ Le CSV doit contenir une colonne 'sms'")
        else:
            df["cleaned"] = df["sms"].astype(str).apply(clean_text)
            X = vectorizer.transform(df["cleaned"])
            df["prediction"] = model.predict(X)
            df["class"] = df["prediction"].map({0: "Ham", 1: "SPAM"})

            st.success("Analyse terminée !")
            st.dataframe(df[["sms", "class"]])

            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger Résultats", csv_out, "predictions.csv")
    except Exception as e:
        st.error(f"Erreur: {e}")
