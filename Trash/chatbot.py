import bs4 as bs
import urllib.request
import re
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import csv

thisdict = {}
questions = []

# with open("./datasets/lumosquestiongenerator.csv") as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     line_count = 0
#     for row in csv_reader:
#         if line_count  == 0:
#             print(f'Column names are {",".join(row)}')
#         else:
#             print(row)
#             questions.append(row[1])
#             thisdict[row[1]] = row[2]
#         line_count+=1
labels = []
courses = []
courseName = []

with open("course.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count  == 0:
            labels = row
        else:
            #print(row)
            questions.append(row[0])
            course = {
                labels[0] : row[0],
                labels[1] : row[1],
                labels[2] : row[2],
                labels[3] : row[3],
                labels[4] : row[4],
                labels[5] : row[5],
                labels[6] : row[6]
            }
            courses.append(course)
            thisdict[row[0]] = course
            
        line_count+=1

print(thisdict['Diploma in Accounting']['Fees'])


cat_data = urllib.request.urlopen('https://articles.unienrol.com/pathway-after-spm/').read()

#Find all the paragraph html from the web page

cat_data_paragraphs = bs.BeautifulSoup(cat_data, features="html.parser").find_all('p')

#Creating lower text corpus of cat paragraphs
cat_text = ''

#Creating lower text corpus of cat paragraphs
for p in cat_data_paragraphs:
    cat_text += p.text.lower()

cat_text = re.sub(r'\s+', ' ',re.sub(r'\[[0-9]*\]', ' ', cat_text))
cat_sentences = nltk.sent_tokenize(cat_text)
questions_text = ''.join(questions)
# questions_text = re.sub(r'\s+', ' ',re.sub(r'\[[0-9]*\]', ' ', questions_text))
# question_sentences = nltk.sent_tokenize(questions_text)

def checkQuestion_similarity(user_query):
    questions.append(user_query)

    vectorizer = TfidfVectorizer()
    sentences_vectors = vectorizer.fit_transform(questions)
    vector_values = cosine_similarity(sentences_vectors[-1], sentences_vectors)

    similarQuestion = questions[vector_values.argsort()[0][-2]]

    input_check = vector_values.flatten()
    input_check.sort()

    if input_check[-2] == 0:
        return "I dun understand what you say, please try again la haiiya"
    else:
        if 'branch' in user_query:
            return thisdict[similarQuestion]['branch']
        else:
            return thisdict[similarQuestion]

def chatbot_answer(user_query):

    #Append the query to the sentences list
    cat_sentences.append(user_query)

    #Create the sentences vector based on the list
    vectorizer = TfidfVectorizer()
    sentences_vectors = vectorizer.fit_transform(cat_sentences)

    #Measure    the cosine similarity and take the second closest index because first index is the user query
    vector_values = cosine_similarity(sentences_vectors[-1], sentences_vectors)

    answer = cat_sentences[vector_values.argsort()[0][-2]]

    input_check = vector_values.flatten()
    input_check.sort()

    if input_check[-2] == 0:
        return "I dun understand what you say, please try again la haiiya"
    else:
        return answer

print("Hello, I am the Cat Chatbot. What is your meow questions?:")
while(True):
    query = input().lower()
    if query not in ['bye', 'good bye', 'take care']:
        print("Cat Chatbot: ", end="")
        print(checkQuestion_similarity(query))
        questions.remove(query)
    else:
        print("See You Again")
        break
