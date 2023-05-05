from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
data = [
    ("book a flight from london to paris", "flight_booking"),
    ("play some music", "music_player"),
    ("set an alarm for 6 am", "alarm_setter"),
    ("what's the weather like today?", "weather_query")
]

# Split data into training and testing sets
x = [i[0] for i in data] # input data
y = [i[1] for i in data] # labels
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Vectorize input data
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train SVM model
clf = LinearSVC()
clf.fit(X_train_vec, y_train)

# Evaluate SVM model
y_pred = clf.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

# Use SVM model for prediction
test_input = "gonna sleep at 7 am"
test_input_vec = vectorizer.transform([test_input])
prediction = clf.predict(test_input_vec)
print("Prediction: ", prediction[0])
