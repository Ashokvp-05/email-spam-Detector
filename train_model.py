import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only necessary columns
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(df['text'])
y = df['label']

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Save Model
with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save Vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model training completed. Files saved successfully!")
