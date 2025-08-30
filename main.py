# Sentiment Analysis Project
# A machine learning example: classify movie reviews as positive or negative


import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#  Custom stopword list
stop_words = {
    "the","is","a","an","and","or","in","on","at","of","to",
    "for","with","by","this","that","it","was","i","you","he",
    "she","they","we","am","are","be","been","being","have",
    "has","had","do","does","did","but","not"
}

#  Dataset (20 short reviews)
data = {
    "text": [
        "I love this movie, it is amazing!",
        "This film was terrible and boring.",
        "What a fantastic experience, truly great.",
        "I hated it, worst movie ever.",
        "Absolutely wonderful, I enjoyed it a lot.",
        "Awful plot, bad acting, do not watch.",
        "Brilliant film, the actors were outstanding.",
        "Poorly written and badly directed.",
        "One of the best movies I have seen.",
        "Terrible waste of time, avoid this film.",
        "Beautiful story and excellent acting.",
        "Disappointing and forgettable.",
        "A masterpiece with amazing visuals.",
        "Predictable and dull.",
        "Loved every minute of it.",
        "I regret watching this film.",
        "Outstanding performance by the cast.",
        "Weak story and poor dialogue.",
        "Highly recommended, a must watch!",
        "Not worth the hype, very bad."
    ],
    "label": [
        "pos", "neg", "pos", "neg", "pos",
        "neg", "pos", "neg", "pos", "neg",
        "pos", "neg", "pos", "neg", "pos",
        "neg", "pos", "neg", "pos", "neg"
    ]
}

df = pd.DataFrame(data)

#  Clean text (lowercase, remove punctuation, stopwords)
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation/numbers
    text = " ".join([word for word in text.split() if word not in stop_words])  # remove stopwords
    return text

df['clean_text'] = df['text'].apply(clean_text)

#  Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.25, random_state=42)

#  TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#  Train models (Naive Bayes + Logistic Regression)
nb_model = MultinomialNB()
lr_model = LogisticRegression(max_iter=1000)

nb_model.fit(X_train_vec, y_train)
lr_model.fit(X_train_vec, y_train)

#  Predictions
nb_pred = nb_model.predict(X_test_vec)
lr_pred = lr_model.predict(X_test_vec)

#  Evaluation
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))

print("\nClassification Report (Logistic Regression):\n", classification_report(y_test, lr_pred))

#  Confusion Matrix (Logistic Regression)
cm = confusion_matrix(y_test, lr_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['neg','pos'], yticklabels=['neg','pos'])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
