import bs4 as bs
import urllib.request
import re
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


df = pd.read_csv('datasets/course.csv')
questions = df.iloc[:, 0].tolist()
print(questions)

def checkCourseName_similarity(user_query):
    questions.append(user_query)

    vectorizer = TfidfVectorizer()
    sentences_vectors = vectorizer.fit_transform(questions)
    vector_values = cosine_similarity(sentences_vectors[-1], sentences_vectors)

    similarQuestion = questions[vector_values.argsort()[0][-2]]

    input_check = vector_values.flatten()
    input_check.sort()

    if input_check[-2] == 0:
        return None
    else:
        questions.remove(user_query)
        return similarQuestion


