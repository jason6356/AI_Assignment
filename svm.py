from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import re

df = pd.read_csv("book1.csv")

focus_general_patterns = {
    'admission requirement': r'.*\b(admission requirement|admission requirements)\b.*',
    'history': r'.*\b(history)\b.*',
    'popular course': r'.*\b(popular course|popular courses)\b.*',
    'career prospect': r'.*\b(career prospect|career prospects)\b.*',
    'financial aid': r'.*\b(financial aid)\b.*',
    'extracurricular': r'.*\b(extracurricular|extracurricular activities)\b.*',
    'course offer': r'.*\b(course offer|course offers)\b.*',
    'admission process': r'.*\b(admission process)\b.*',
    'support service': r'.*\b(support service|support services)\b.*',
    'student housing': r'.*\b(student housing)\b.*',
    'cost': r'.*\b(cost)\b.*',
    'academic support': r'.*\b(academic support)\b.*',
    'career service': r'.*\b(career service|career services)\b.*'

}

X = df['question']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#convert data into numerical 
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

#train the data
clf = LinearSVC()
clf.fit(X_train_counts, y_train)

y_pred = clf.predict(X_test_counts)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

user_input = input("How may I help you?")
user_input_counts = vectorizer.transform([user_input])
predicted_label = clf.predict(user_input_counts)
# print(predicted_label)

focus_category = []
answer = ''

if predicted_label == 'general':
    for focus, pattern in focus_general_patterns.items():
        if re.match(pattern, user_input, re.IGNORECASE):
            focus_category.append(focus)

answer = df.loc[df['focus'].isin(focus_category)]['answer'].values[0]

print(answer)

# Training data
# train_data = [
#     ('What course offer?','course_title'),
#     ('What course offer in KL campus?','course_title','course_location'),
#     ('What is the program overview?', 'program_overview'),
#     ('How much is the course fee?', 'course_fee'),
#     ('Where is the campus located?', 'campus_location'),
#     ('When does the course intake start?', 'course_intake'),
#     ('How long is the course duration?', 'course_duration'),
#     ('Which campus offer this course?','campus_location','course_title'),
#     ('Where is the Computer Science course located?', 'course_location'),
#     ('What is the title of the Business Administration course?', 'course_title'),
#     ('Can you provide an overview of the Nursing program?', 'program_overview'),
#     ('What is the intake period for the Marketing course?', 'course_intake'),
#     ('How long is the Engineering course duration?', 'course_duration'),
#     ('What is the location of the Law School campus?', 'campus_location'),
#     ('What is the fee for the Psychology course?', 'course_fee'),
#     ('What is the duration of the Graphic Design course?', 'course_duration'),
#     ('What is the Computer Engineering program about?', 'program_overview'),
#     ('What is the fee for the Accounting course?', 'course_fee'),
#     ('Which campus offers the Civil Engineering program?', 'campus_location'),
#     ('What is the intake period for the Marketing course?', 'course_intake'),
#     ('Where is the Fine Arts course located?', 'course_location'),
#     ('What is the course title of the Mathematics program?', 'course_title'),   
#     ('What is the duration of the Architecture course?', 'course_duration'),
#     ('What is the fee for the Information Technology course?', 'course_fee'),
#     ('Can you provide an overview of the Chemistry program?', 'program_overview'),
#     ('What is the location of the Biology campus?', 'campus_location'),
#     ('What is the course title of the Journalism program?', 'course_title'),
#     ('What is the intake period for the Public Relations course?', 'course_intake'),
#     ('Where is the Physics course located?', 'course_location')
# ]
# x = [i[0] for i in train_data] # input data
# y = [i[1] for i in train_data] # labels
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Preprocess and extract features from the training data
# vectorizer = TfidfVectorizer()
# X_train = vectorizer.fit_transform([x[0] for x in train_data])
# y_train = [x[1] for x in train_data]

# # Train the SVM model
# svm_model = SVC(kernel='linear')
# svm_model.fit(X_train, y_train)

# # Predict the intent of new user input
# user_input = input("How may I help you?:")
# X_test = vectorizer.transform([user_input])
# y_pred = svm_model.predict(X_test)

# # Map the predicted intent to the appropriate response
# if y_pred == 'program_overview':
#     response = 'The program overview is...'
# elif y_pred == 'cpurse_title':
#     response = 'The course offer is...'
# elif y_pred == 'course_fee':
#     response = 'The course fee is...'
# elif y_pred == 'campus_location':
#     response = 'The campus is located at...'
# elif y_pred == 'course_intake':
#     response = 'The course intake on...'
# elif y_pred == 'course_duration':
#     response = 'The course duration is...'
# else:
#     response = 'I am sorry I do not understand'
         
            
# print(response)