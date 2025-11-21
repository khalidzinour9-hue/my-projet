import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ====================== Télécharger ressources NLTK ======================
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# ====================== Charger le modèle ======================
with open("nltk_spam_model.pkl", "rb") as f:
    model = pickle.load(f)

# ====================== Préparation du texte ======================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def nettoyer_texte(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return tokens

def extraire_features(mots):
    return {mot: True for mot in mots}

# ====================== CSS PRO ======================
css = """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1e3c72, #2a5298, #6f00ff, #4b6cb7);
    background-size: 400% 400%;
    animation: gradientBG 20s ease infinite;
    color: white;
}
@keyframes gradientBG {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}
[data-testid="stAppViewContainer"] {backdrop-filter: blur(3px);}
h1, label, p { font-family: 'Poppins', sans-serif; color: white !important; text-align:center; }
.card {
    background: rgba(255,255,255,0.10);
    padding: 30px; border-radius:20px;
    border:1px solid rgba(255,255,255,0.25);
    backdrop-filter: blur(15px);
    margin-top:30px;
}
input { background: rgba(255,255,255,0.18) !important; color:black !important; border-radius:12px !important; padding:10px; }
.stButton>button {
    background: rgba(255,255,255,0.25);
    color:white; padding:12px 25px; border-radius:12px;
    border:1px solid rgba(255,255,255,0.4); font-weight:bold;
    transition:0.3s;
}
.stButton>button:hover { background: rgba(255,255,255,0.35); transform: scale(1.07); }
.result {
    background: rgba(0,0,0,0.45); padding:20px; color:#fff; font-size:18px;
    border-radius:12px; border-left:4px solid #ff5252; margin-top:15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.result.ham { border-left:4px solid #4caf50; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ====================== Interface ======================
st.markdown("<h1>🔍 Détecteur de Spam</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#e0e0e0;'>Analysez si votre message est Spam ou Ham</p>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

# Formulaire pour fiabilité
with st.form(key='spam_form'):
    message = st.text_input("Entrez le message à analyser :")
    submit_button = st.form_submit_button(label='Analyser')

    if submit_button:
        if message.strip() == "":
            st.error("❌ Veuillez entrer un message.")
        else:
            tokens = nettoyer_texte(message)
            features = extraire_features(tokens)
            pred = model.classify(features)

            if pred.lower() == "spam":
                st.markdown("<div class='result'>🚨 Le message est SPAM </div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result ham'>✔ Le message est HAM </div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
