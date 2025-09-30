import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud

# Custom stopword list
stop_words = {
    "the","is","a","an","and","or","in","on","at","of","to",
    "for","with","by","this","that","it","was","i","you","he",
    "she","they","we","am","are","be","been","being","have",
    "has","had","do","does","did","but","not"
}

# Load dataset
data = []
with open("reviews.txt", "r") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        text, label = line.rsplit(",", 1)
        data.append({"text": text, "label": label})

df = pd.DataFrame(data)

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Plot class distribution
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
nb_model.fit(X_train_vec, y_train)
nb_pred = nb_model.predict(X_test_vec)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
lr_pred = lr_model.predict(X_test_vec)

svm_model = LinearSVC()
svm_model.fit(X_train_vec, y_train)
svm_pred = svm_model.predict(X_test_vec)

# -----------------------------
# Cross-Validation Accuracy
# -----------------------------
print("Cross-Validation Accuracy (5-fold):")
for name, model in zip(['Naive Bayes', 'Logistic Regression', 'SVM'],
                       [nb_model, lr_model, svm_model]):
    scores = cross_val_score(model, vectorizer.transform(df['clean_text']), df['label'], cv=5)
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

# Evaluate models
print("\nTest Set Accuracy:")
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

print("\nClassification Report (Logistic Regression):\n",
      classification_report(y_test, lr_pred, zero_division=0))
print("Classification Report (SVM):\n",
      classification_report(y_test, svm_pred, zero_division=0))

# Plot confusion matrices
models_preds = {'Logistic Regression': lr_pred, 'SVM': svm_pred}
for name, pred in models_preds.items():
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['neg','pos'], yticklabels=['neg','pos'])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Word clouds
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

# Top predictive words for Logistic Regression
feature_names = vectorizer.get_feature_names_out()
coef_lr = lr_model.coef_[0]
top_pos_lr = sorted(zip(coef_lr, feature_names))[-10:]
top_neg_lr = sorted(zip(coef_lr, feature_names))[:10]

print("\nTop Positive Words (Logistic Regression):", [w for c, w in top_pos_lr])
print("Top Negative Words (Logistic Regression):", [w for c, w in top_neg_lr])

# Top predictive words for SVM
coef_svm = svm_model.coef_[0]
top_pos_svm = sorted(zip(coef_svm, feature_names))[-10:]
top_neg_svm = sorted(zip(coef_svm, feature_names))[:10]

print("\nTop Positive Words (SVM):", [w for c, w in top_pos_svm])
print("Top Negative Words (SVM):", [w for c, w in top_neg_svm])

# -----------------------------
# Interactive review prediction with ensemble voting
# -----------------------------
def ensemble_predict(vec):
    preds = [nb_model.predict(vec)[0],
             lr_model.predict(vec)[0],
             svm_model.predict(vec)[0]]
    return Counter(preds).most_common(1)[0][0]

while True:
    user_input = input("\nEnter a movie review (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    clean_input = clean_text(user_input)
    input_vec = vectorizer.transform([clean_input])

    nb_prediction = nb_model.predict(input_vec)[0]
    lr_prediction = lr_model.predict(input_vec)[0]
    svm_prediction = svm_model.predict(input_vec)[0]
    ensemble_prediction = ensemble_predict(input_vec)

    print("\nPredicted Sentiments:")
    print("Naive Bayes:", nb_prediction.upper())
    print("Logistic Regression:", lr_prediction.upper())
    print("SVM:", svm_prediction.upper())
    print("Ensemble Voting:", ensemble_prediction.upper())
