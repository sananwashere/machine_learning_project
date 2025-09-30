import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud

# Custom stopword list
stop_words = {
    "the","is","a","an","and","or","in","on","at","of","to",
    "for","with","by","this","that","it","was","i","you","he",
    "she","they","we","am","are","be","been","being","have",
    "has","had","do","does","did","but","not"
}

# Load dataset from .txt file
data = []
with open("reviews.txt", "r") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
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

# Class distribution
plt.figure(figsize=(5,4))
sns.countplot(x=df['label'])
plt.title("Sentiment Distribution")
plt.show()

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

# Predictions
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

# Word Clouds
pos_text = " ".join(df[df['label']=="pos"]['clean_text'])
neg_text = " ".join(df[df['label']=="neg"]['clean_text'])

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(WordCloud(width=400, height=300, background_color="white").generate(pos_text))
plt.axis("off")
plt.title("Positive Word Cloud")

plt.subplot(1,2,2)
plt.imshow(WordCloud(width=400, height=300, background_color="white").generate(neg_text))
plt.axis("off")
plt.title("Negative Word Cloud")
plt.show()

# Top Predictive Words (Logistic Regression)
feature_names = vectorizer.get_feature_names_out()
coef = lr_model.coef_[0]

top_pos = sorted(zip(coef, feature_names))[-10:]
top_neg = sorted(zip(coef, feature_names))[:10]

print("\nTop Positive Words:", [w for c, w in top_pos])
print("Top Negative Words:", [w for c, w in top_neg])

# Interactive review prediction
while True:
    user_input = input("\nEnter a movie review (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    clean_input = clean_text(user_input)
    input_vec = vectorizer.transform([clean_input])
    prediction = lr_model.predict(input_vec)[0]
    print("Predicted sentiment:", prediction.upper())
