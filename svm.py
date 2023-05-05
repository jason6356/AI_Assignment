from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Training data
train_data = [
    ('What course offer?','course_title'),
    ('What course offer in KL campus?','course_title','course_location'),
    ('What is the program overview?', 'program_overview'),
    ('How much is the course fee?', 'course_fee'),
    ('Where is the campus located?', 'campus_location'),
    ('When does the course intake start?', 'course_intake'),
    ('How long is the course duration?', 'course_duration'),
    ('Which campus offer this course?','campus_location','course_title'),
    ('Where is the Computer Science course located?', 'course_location'),
    ('What is the title of the Business Administration course?', 'course_title'),
    ('Can you provide an overview of the Nursing program?', 'program_overview'),
    ('What is the intake period for the Marketing course?', 'course_intake'),
    ('How long is the Engineering course duration?', 'course_duration'),
    ('What is the location of the Law School campus?', 'campus_location'),
    ('What is the fee for the Psychology course?', 'course_fee'),
    ('What is the duration of the Graphic Design course?', 'course_duration'),
    ('What is the Computer Engineering program about?', 'program_overview'),
    ('What is the fee for the Accounting course?', 'course_fee'),
    ('Which campus offers the Civil Engineering program?', 'campus_location'),
    ('What is the intake period for the Marketing course?', 'course_intake'),
    ('Where is the Fine Arts course located?', 'course_location'),
    ('What is the course title of the Mathematics program?', 'course_title'),   
    ('What is the duration of the Architecture course?', 'course_duration'),
    ('What is the fee for the Information Technology course?', 'course_fee'),
    ('Can you provide an overview of the Chemistry program?', 'program_overview'),
    ('What is the location of the Biology campus?', 'campus_location'),
    ('What is the course title of the Journalism program?', 'course_title'),
    ('What is the intake period for the Public Relations course?', 'course_intake'),
    ('Where is the Physics course located?', 'course_location')
]
x = [i[0] for i in train_data] # input data
y = [i[1] for i in train_data] # labels
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Preprocess and extract features from the training data
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform([x[0] for x in train_data])
y_train = [x[1] for x in train_data]

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predict the intent of new user input
user_input = input("How may I help you?:")
X_test = vectorizer.transform([user_input])
y_pred = svm_model.predict(X_test)

# Map the predicted intent to the appropriate response
if y_pred == 'program_overview':
    response = 'The program overview is...'
elif y_pred == 'cpurse_title':
    response = 'The course offer is...'
elif y_pred == 'course_fee':
    response = 'The course fee is...'
elif y_pred == 'campus_location':
    response = 'The campus is located at...'
elif y_pred == 'course_intake':
    response = 'The course intake on...'
elif y_pred == 'course_duration':
    response = 'The course duration is...'
else:
    response = 'I am sorry I do not understand'
         
            
print(response)