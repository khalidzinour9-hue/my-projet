# Spam Detector - Mini Project

Project structure:
- train_model.py  -> script to train a logistic regression model and save `spam_model.pkl` and `tfidf.pkl`.
- app.py          -> Streamlit app to input a message and predict Spam/Ham.
- spam.csv        -> small sample dataset included for quick test (replace with full SMS Spam Collection Dataset for better results).
- Mini projet NLP- 5.pdf -> project brief (copied from your upload).

Quick start:
1. (Optional) create a virtualenv and install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the model (this will create `spam_model.pkl` and `tfidf.pkl`):
   ```
   python train_model.py
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

Notes:
- The included `spam.csv` is a tiny sample for testing. For full project use the full SMS Spam Collection Dataset.
- If you run into NLTK stopwords error, run in Python:
   ```python
   import nltk
   nltk.download('stopwords')
   ```
