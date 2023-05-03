from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset of user queries and their corresponding intents
data = pd.read_csv("data.csv")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data["text"], data["intent"], test_size=0.2, random_state=42)

# Vectorize input data
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes model
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Evaluate Naive Bayes model
y_pred = clf.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

# Use Naive Bayes model for prediction
user_input = "I want to book a flight to New York"
user_input_vec = vectorizer.transform([user_input])
intent = clf.predict(user_input_vec)
print("User intent: ", intent[0])
