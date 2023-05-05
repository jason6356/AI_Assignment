import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
from keras.models import load_model

import svm
import nlp
import naiveBayes

import setupData

labels, courses, courseName, courseDictionary = setupData.init()

print(labels)

# Load data
df = pd.read_csv('datasets/dataset.csv')


#Create an object of WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#Import the GL Bot corpus file for preprocessing
words=[]
classes = labels
documents = []
ignore_words = ['?', '!']
#   firstaid_file = open("firstaid.json").read()
#intents = json.loads(firstaid_file)

#nltk.download('punkt')
#nltk.download('wordnet')

for index,row in df.iterrows():

    data = row['Data']
    intent = row['Intent']

    w = nltk.word_tokenize(data)
    words.extend(w)
    #print(f"Data: {data}, Intent: {intent}")
    documents.append((w,intent))

#Lemmatize, lowercase each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

#Sort the words by classes
classes = sorted(list(set(classes)))

#Documents = combination between patterns and intents
print (len(documents), "documents")

#Classes = intents
print (len(classes), "classes", classes)

#Words = all words, vocabulary
print (len(words), "unique lemmatized words", words)

#Creating a pickle file to store the Python objects which we will use while predicting
pickle.dump(words,open('words.pkl','wb')) 
pickle.dump(classes,open('classes.pkl','wb'))


# In[7]:


#Create the training data
training = []

#Create an empty array for the output
output_empty = [0] * len(classes)

#Training set, bag of words for each sentence
for doc in documents:
    
    #Initialize our bag of words
    bag = []
    
    #List of tokenized words for the pattern
    pattern_words = doc[0]
   
    #Lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    #Create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    #Output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

#Shuffle features and converting it into numpy arrays
random.shuffle(training)
training = np.array(training)

#Create train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])

print("Training data has been created")


# ### Modeling

# In[8]:


#Create Neural Network model to predict the responses
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))


# In[9]:


#Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
#sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

from tensorflow.keras.optimizers import SGD

#Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# In[10]:


#Fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=10, verbose=1)
model.save('chatbot.h5', hist) #We will pickle this model to use in the future
print("\n")
print("*"*50)
print("\nModel Created Successfully!")


# In[11]:


#Load the saved model file
model = load_model('chatbot.h5')
#intents = json.loads(open("firstaid.json").read())
df = pd.read_csv("datasets/course.csv")
intents = df.to_dict(orient='records')
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


# ### Deployment

# In[12]:


def clean_up_sentence(sentence):

    #Tokenize the pattern > split words into array
    sentence_words = nltk.word_tokenize(sentence)
    
    #Stem each word > create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


#Return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):

    #Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)

    #Bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
               
                # Assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
   
    #Filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    error = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>error]
    
    #Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return classes[r[0]]


# In[13]:


#Function to get the response from the model
#Deprecated in favor of get_response

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

#Function to predict the class and get the response

def chatbot_response(text):

    isGeneralText = svm.SVM_getIntent(text)

    if isGeneralText == 'specific':
        ints = predict_class(text, model)
        #ints = naiveBayes.predict_intent_with_nb(text)
        print(ints)
        coursePrediction = nlp.checkCourseName_similarity(text)

        if coursePrediction is None:
            return "Unable to identify which course are you refering to!"

        return courseDictionary[coursePrediction][ints]
    else:
        return isGeneralText


    


# In[14]:


#Fucntion to rate the service before end conversation

def rateService():
    print("-"*50)
    print("\n"+ "Thank you for using our AI chatbot. It would be great if you could rate TARUMT chatbot." + "\n")
    print("1 star  - Very unsatisfied")
    print("2 stars - Unsatisfied")
    print("3 stars - Neutral")
    print("4 stars - Satisdfied")
    print("5 stars - Very satisfied")
    
    rate = int(input("Your rating (1-5) > "))
    while rate <= 0 or rate > 5:
        print("Invalid! Please enter 1 to 5 only.")
        rate = int(input("Your rating (1-5) > "))
    if rate == 1 or rate == 2:
        print("Sorry for the bad experience :( Let's hear us your feedback, it is valuable to us.")
        feedback = str(input("Feedback: "))
        print("Thank you for your feedback. We will use your feedback for future improvement.")
    else:
        print("We really appreciate you taking the time to share your rating with us, thank you! :)")


# In[15]:


#Function to start the chat bot which will continue until the user type 'end' or 'bye'

def startChat():
    print("TARUMT Bot: Hello! I am TARUMT, your personal AI course offer assistant.")
    name = str(input("TARUMT Bot: What is your name ? > "))
    print("TARUMT Bot: Nice to meet you, " + (name) + "!")
    print("-"*50)
    while True:
        inp = str(input((name)+": "))
        if inp.lower()=="end" or inp.lower() =="bye":
            print("TARUMT Bot: Bye " + (name) + "!")
            rateService()
            break
        if inp.lower()== '' or inp.lower()== '*': #If user empty input
            print('Please enter again!')
            print("-"*50)
        else:
            print(f"TARUMT Bot: {chatbot_response(inp)}"+'\n')
            print("-"*50)


# In[17]:


#Start the chat bot
startChat()


# In[ ]:




