import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from sklearn.metrics import accuracy_score, f1_score
from nltk import NaiveBayesClassifier
import pickle

# Télécharger les ressources NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# 1. Charger le dataset
df = pd.read_csv("spam (or) ham.csv", encoding="latin-1")
df = df.iloc[:, :2]
df.columns = ['label','message']

# 2. Nettoyage du texte
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def nettoyer_texte(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return tokens

df['tokens'] = df['message'].apply(nettoyer_texte)

# 3. Conversion tokens -> dict (features)
def extraire_features(mots):
    return {mot: True for mot in mots}

df['features'] = df['tokens'].apply(extraire_features)

# 4. Split train/test
data = list(zip(df['features'], df['label']))
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# 5. Entraînement
classifier = NaiveBayesClassifier.train(train_data)

# 6. Evaluation
y_true = [label for (_, label) in test_data]
y_pred = [classifier.classify(feat) for (feat, _) in test_data]

print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1-score:", f1_score(y_true, y_pred, average='macro'))

# 7. Sauvegarde du modèle
with open("spam_model.pkl", "wb") as f:
    pickle.dump(classifier, f)

print("✅ Modèle sauvegardé: spam_model.pkl")
