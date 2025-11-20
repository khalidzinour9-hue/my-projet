import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    import nltk
# هذا السطر سيضمن تنزيل مورد 'stopwords' عند تشغيل التطبيق
# إذا كان بالفعل موجودًا، فلن يقوم بتنزيله مرة أخرى.
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# الآن الكود يمكنه العمل:
from nltk.corpus import stopwords
# ...
# stop_words = set(stopwords.words('english'))
    stop_words = set(stopwords.words('english'))
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

st.title("📩 Détecteur de Spam")
st.write("Prédisez si le message est un Spam ou un message Ham.")

message = st.text_area("✉️ Entrez votre message ici :")

try:
    model = pickle.load(open("spam_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf.pkl", "rb"))
except FileNotFoundError:
    st.warning("⚠️ Les fichiers du modèle (spam_model.pkl / tfidf.pkl) sont introuvables. Veuillez exécuter `train_model.py` pour les générer.")
    model = None
    vectorizer = None

if st.button("🔍 Analyser"):
    if not message.strip():
        st.warning("⚠️ Veuillez entrer un message valide.")
    else:
        if model is None or vectorizer is None:
            st.error("❌ Fichiers du modèle manquants. Lancez `train_model.py` pour les créer.")
        else:
            cleaned = clean_text(message)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            if pred == 1:
                st.error("🔴 Ce message est un SPAM.")
            else:
                st.success("🟢 Ce message est HAM .")
