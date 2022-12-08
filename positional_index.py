import numpy as np
import pandas as pd
import json
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from string import punctuation
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel
import os
from nltk.corpus import stopwords


def load_index():
    with open('index.json', 'r') as f:
        index = json.load(f)
    return index


index=load_index()
#search positional index for a query
def search_index(query, index):
    query = word_tokenize(query)
    query = [word for word in query if word not in stopwords.words('english')]
    query = [word for word in query if word not in punctuation]
    query = [word for word in query if not word.isdigit()]
    query = [word.lower() for word in query]
    query = list(set(query))
    postings = [index[word] for word in query]
    intersection = set(postings[0])
    for posting in postings:
        intersection = intersection.intersection(posting)
    return intersection

print(search_index('dune',index))


