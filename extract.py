import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize, word_tokenize

import numpy as np

nltk.download('punkt')

STOPLST = stopwords.words('english') + stopwords.words('russian')


def get_text(filename):
    with open(filename, encoding='utf-8') as f:
        text = f.read()
    return text


def preprocess(sentences: list) -> list:
    return [[word.lower() for word in word_tokenize(sentence) if word.isalpha()
             and word not in STOPLST] for sentence in sentences]


def similarity(sent1, sent2):
    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        vector1[all_words.index(w)] += 1

    for w in sent2:
        vector2[all_words.index(w)] += 1

    return cosine_distance(vector1, vector2)


def build_matrix(sentences):
    sents = preprocess(sentences)
    size = len(sents)
    mat = np.zeros((size, size))
    for i in range(size):
        for j in range(i+1, size):
            sim = similarity(sents[i], sents[j])
            mat[i, j] = sim
            mat[j, i] = sim
    return mat


def get_pagerank(mat, top_k):
    importance = np.mean(mat, axis=0)
    idx = importance.argsort()[:top_k]
    idx.sort()
    return idx


def summarize_text(text: str, top_k: int = 20) -> str:
    sentences = sent_tokenize(text)[:1000]
    mat = build_matrix(sentences)
    idx = get_pagerank(mat, top_k)
    return '\n'.join(sentences[i] for i in idx)
