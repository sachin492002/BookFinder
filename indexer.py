import pickle

import numpy as np
import pandas as pd
import json
from nltk.tokenize import word_tokenize
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from string import punctuation
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel
import os
from nltk.corpus import stopwords
from nltk.stem.porter import *


# load data from books_enriched.csv


def load_data():
    df = pd.read_csv('books_enriched.csv')
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    df['authors'] = df['authors'].apply(lambda x: x.replace(
        '[', '').replace(']', '').replace("'", ""))
    df['genres'] = df['genres'].apply(lambda x: x.replace(
        '[', '').replace(']', '').replace("'", ""))

    # df['description'] = df['description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
    return df


def build_positional_index(df):
    df['original_publication_year'] = df['original_publication_year'].map(str)
    index = {}
    df['all'] = df['all'] = df['authors'] + ' ' + df['description'] + ' ' + \
                            df['genres'] + ' ' + df['original_title'] + \
                            ' ' + df['original_publication_year']
    # clean all symbols in all
    df['all'] = df['all'].apply(lambda x: ''.join(
        [i for i in x if i not in frozenset(punctuation)]))
    # stem all
    stemmer = PorterStemmer()
    df['all'] = df['all'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
    all = df['all'].tolist()
    for i in range(len(all)):
        for word in str(all[i]).split():
            if word not in index:
                index[word] = {i: [1]}
            else:
                if i not in index[word]:
                    index[word][i] = [1]
                else:
                    index[word][i].append(index[word][i][-1] + 1)
    return index


def store_index(index):
    with open('index.json', 'w') as f:
        json.dump(index, f)
    return


df = load_data()
# p1 = build_positional_index(df)
# store_index(p1)


def build_tfidf(df):
    df['original_publication_year'] = df['original_publication_year'].map(str)
    df['all'] = df['authors'] + ' ' + df['description'] + ' ' + df['genres'] + \
                ' ' + df['original_title'] + ' ' + df['original_publication_year']
    all = df['all'].tolist()
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(all)
   
    return tfidf_matrix, tfidf


# tf_idf, vectorize = build_tfidf(df)

# print(type(tf_idf),type(vectorize))

def store_tfidf(tfidf_matrix, vectorize):
    sparse.save_npz("tfidf_matrix.npz", tfidf_matrix)
    pickle.dump(vectorize, open('vectorize.pickle', 'wb'))


# store_tfidf(tf_idf, vectorize)
#load tf_idf and vectrize
def load_tfidf():
    tfidf_matrix = sparse.load_npz("tfidf_matrix.npz")
    vectorize = pickle.load(open('vectorize.pickle', 'rb'))
    return tfidf_matrix, vectorize




# loading tfidf matrix and vectorizer
tf_idf, vectorize = load_tfidf()


# return top 10 index for query using cosine similarity
def search(query, tfidf_matrix, df):
    query = [query]
    query_vec = vectorize.transform(query).toarray()
    cosine_similarities = linear_kernel(query_vec, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-11:-1]
    return related_docs_indices


def eachTup(tfidf_matrix,df):
    df['original_publication_year'] = df['original_publication_year'].map(str)
    df['all'] = df['authors'] + ' ' + df['description'] + ' ' + df['genres'] + \
                ' ' + df['original_title'] + ' ' + df['original_publication_year']
    all = df['all'].tolist()
    indx=1
    js_cos={}
    for i in df['all']:
        i=[i]
        jv= vectorize.transform(i).toarray()
        cosine_similarities = linear_kernel(jv, tfidf_matrix).flatten()
        #store cosine similarities in sorted order
        sorted_cosine = cosine_similarities.argsort()[:-100:-1]
        js_cos[indx]=sorted_cosine
        indx+=1
    
    try:
        geeky_file = open('tupleCosine', 'wb')
        pickle.dump(js_cos, geeky_file)
        geeky_file.close()
    except:
        print("Something went wrong")

with open('tupleCosine', 'rb') as handle:
    unserialized_data = pickle.load(handle)


#feedback
def topcosine(data,rel_vector,df1,prevvious):
    data=[data]
    li=[]
    r=0
    for i in prevvious:
        if(rel_vector[r]==1):
            li.append(i)
            for j in unserialized_data[i]:
                li.append(j)
        r+=1
    df=df1.iloc[li]
    df['original_publication_year'] = df['original_publication_year'].map(str)
    df['all'] = df['authors'] + ' ' + df['description'] + ' ' + df['genres'] + \
                ' ' + df['original_title'] + ' ' + df['original_publication_year']
    all = df['all'].tolist()
    tfidf = TfidfVectorizer()
    tfidf_matrix = vectorize.transform(all)
    newjv=vectorize.transform(data).toarray()
    cosine_similarities = linear_kernel(newjv, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-11:-1]
    
    return related_docs_indices,df1

    
def feedsearch(query,previous,df):
    indexes,newdf=topcosine(query,np.array([1,0,1,0,0,1,0,0,1,1]),df,search(query,tf_idf,df))
    objArr = []

    for i in indexes:
        obj = {}
        obj['title'] = newdf['original_title'][i]
        obj['author'] = newdf['authors'][i]
        obj['genre'] = newdf['genres'][i]
        obj['year'] = newdf['original_publication_year'][i]
        obj['index']=i
        obj['description'] = newdf['description'][i]
        obj['image'] = newdf['image_url'][i]
        objArr.append(obj)
    return objArr



print(feedsearch("dune",search("dune",tf_idf,df),df))

# indexes = search('dune', tf_idf, df)
# for i in indexes:
#     print(i, df['original_title'][i])


def searchFlask(query):

    indexes = search(query, tf_idf, df)
    objArr = []

    for i in indexes:
        obj = {}
        obj['title'] = df['original_title'][i]
        obj['author'] = df['authors'][i]
        obj['genre'] = df['genres'][i]
        obj['year'] = df['original_publication_year'][i]
        obj['index']=i
        obj['description'] = df['description'][i]
        obj['image'] = df['image_url'][i]
        objArr.append(obj)
    return objArr



#evaluate
q = "i was roaming around"
totalReleventDoc = 5
ans = search(q, tf_idf, df)
releventDoc = [1, 2, 3, 6, 10] 
rank = 1
for inded in ans:
    if rank in releventDoc:
        print("Rank : ", rank, ", docID : ", inded, ", relevant doc")
    else:
        print("Rank : ", rank, ", docID : ", inded, ", Not relevant doc")
    rank = rank + 1

recall = []
precision = []

rank = 1
relevalntRetrived = 0
for docID in ans:
    if rank in releventDoc:
        relevalntRetrived = relevalntRetrived + 1
    recall.append(relevalntRetrived/totalReleventDoc)
    precision.append(relevalntRetrived/rank)
    print("Rank : ", rank, ", recall : ", recall[rank-1], ", precision : ",precision[rank-1])
    rank = rank + 1
# 11 point interpolation 
# [0.0 , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
precision.reverse()

for i in range(1,len(precision),1):
    precision[i] = max(precision[i-1],precision[i])
precision.reverse()
print(precision)

plt.plot(recall, precision,'bo')
plt.plot(recall, precision, color = "red")
plt.title('precision Vs recall')
plt.xlabel('recall')
plt.ylabel('precision')
plt.show()