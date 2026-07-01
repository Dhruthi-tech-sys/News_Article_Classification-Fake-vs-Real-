import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load cleaned dataset
df = pd.read_csv("data/cleaned_news.csv")

# ADD THIS LINE HERE
print(df.isnull().sum())

# Remove rows with missing values
df = df.dropna(subset=["content"])

# Convert to string
df["content"] = df["content"].astype(str)

print("Dataset loaded!")

# Features and labels
X = df["content"]
y = df["label"]

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_vectorized = tfidf.fit_transform(X)

print("TF-IDF shape:", X_vectorized.shape)

# Save vectorizer
joblib.dump(tfidf, "model/vectorizer.pkl")

print("Vectorizer saved successfully!")