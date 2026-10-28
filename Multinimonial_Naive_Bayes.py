import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("News.csv")

# 2. Separate text and target
X = df["Text"]
y = df["Category"]

# 3. Split the raw text before fitting the vectorizer
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Convert text into word-count features
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# 5. Create and train the Naïve Bayes model
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# 6. Predict test categories
y_pred = model.predict(X_test_counts)

# 7. Calculate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# 8. Predict a new news article
new_article = [
    "The team won the football championship after scoring two goals."
]

new_article_counts = vectorizer.transform(new_article)
prediction = model.predict(new_article_counts)

print("Predicted Category:", prediction[0])