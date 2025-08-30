
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Custom stopword list
stop_words = {
    "the","is","a","an","and","or","in","on","at","of","to",
    "for","with","by","this","that","it","was","i","you","he",
    "she","they","we","am","are","be","been","being","have",
    "has","had","do","does","did","but","not"
}

# Load dataset from .txt file (handles commas in reviews)
data = []
with open("reviews.txt", "r") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue  # skip empty lines
        text, label = line.rsplit(",", 1)  # split only on last comma
        data.append({"text": text, "label": label})

df = pd.DataFrame(data)

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.25, random_state=42
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train models
nb_model = MultinomialNB()
lr_model = LogisticRegression(max_iter=1000)

nb_model.fit(X_train_vec, y_train)
lr_model.fit(X_train_vec, y_train)

# Predictions on test set
nb_pred = nb_model.predict(X_test_vec)
lr_pred = lr_model.predict(X_test_vec)

# Evaluation
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("\nClassification Report (Logistic Regression):\n",
      classification_report(y_test, lr_pred, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, lr_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['neg','pos'], yticklabels=['neg','pos'])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#  Interactive review prediction
while True:
    user_input = input("\nEnter a movie review (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    clean_input = clean_text(user_input)
    input_vec = vectorizer.transform([clean_input])
    prediction = lr_model.predict(input_vec)[0]
    print("Predicted sentiment:", prediction.upper())
