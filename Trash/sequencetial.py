import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import pandas as pd

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
from keras.models import load_model

lemmatizer = WordNetLemmatizer()