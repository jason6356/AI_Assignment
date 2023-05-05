import setupData
import svm
import s2q
import nltk
from nltk.stem import WordNetLemmatizer

labels, courses, courseName, courseDictionary = setupData.init()

train_x, train_y = s2q.nlp()
model,intent,words,classes = s2q.load_model(train_x, train_y)

def clean_up_sentence(sentence):

    lemmatizer = WordNetLemmatizer()

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

def predict_class(sentence, model,words, classes):
   
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
    return return_list

#Function to get the response from the model

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
        print(ints)
        return ints[0]
    else:
        return isGeneralText

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


startChat()