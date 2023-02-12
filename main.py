import matplotlib.pyplot as plt
import csv
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pymorphy2
import re
from datetime import datetime


def cluster_text():
    text = []
    morph = pymorphy2.MorphAnalyzer()
    snowball = SnowballStemmer(language='russian')
    stop_words = stopwords.words('russian')
    with open('data.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in spamreader:
            if i != 0:
                tokens = word_tokenize(row[1], language='russian')
                filtered_tokens = []
                for token in tokens:
                    if token not in stop_words:
                        filtered_tokens.append(token)
                for i in range(len(filtered_tokens)):
                    morped_token = morph.parse(filtered_tokens[i])[0].normal_form
                    filtered_tokens[i] = snowball.stem(morped_token)
                textrow = re.sub('[^А-Яа-я+\sА-Яа-я+]+','',' '.join(sorted(filtered_tokens))).strip()
                if textrow:
                    text.append(textrow)
            i = i+1
    vectorizer = TfidfVectorizer(stop_words={'russian'})
    X = vectorizer.fit_transform(text)
    Sum_of_squared_distances = []
    done = Queue()
    K = range(2, len(set(text)))
    processes = []
    for k in K:
        km = KMeans(n_clusters=k, max_iter=200, n_init=10)
        km = km.fit(X)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    print('How many clusters do you want to use?')
    true_k = int(input())
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
    model.fit(X)

    labels = model.labels_
    clusters = pd.DataFrame(list(zip(text, labels)), columns=['title', 'cluster'])
    # print(clusters.sort_values(by=['cluster']))

    for i in range(true_k):
        print(clusters[clusters['cluster'] == i])

    return

def main():
    cluster_text()