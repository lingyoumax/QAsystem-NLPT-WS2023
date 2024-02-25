import math
import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")


class BM25:
    def __init__(self, corpus: List[List[str]], k1=1.5, b=0.95):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.documents_number = len(corpus)
        self.avgdl = sum(len(document) for document in corpus) / self.documents_number
        self.df = self._calculate_df()
        self.idf = self._calculate_idf()

    def _calculate_df(self):
        df = {}
        for document in self.corpus:
            for word in set(document):
                df[word] = df.get(word, 0) + 1
        return df

    def _calculate_idf(self):
        idf = {}
        for word, freq in self.df.items():
            idf[word] = math.log((self.documents_number - freq + 0.5) / (freq + 0.5) + 1)
        return idf

    def _score(self, document, query):
        score = 0.0
        for word in query:
            if word in self.df:
                idf = self.idf[word]
                term_freq = document.count(word)
                score += (idf * term_freq * (self.k1 + 1)) / (
                        term_freq + self.k1 * (1 - self.b + self.b * len(document) / self.avgdl))
        return score

    def get_scores(self, query):
        scores = []
        for index, document in enumerate(self.corpus):
            score = self._score(document, query)
            scores.append((index, score))
        return scores


def search(bm25, df, query: str, keywords, top_k=3):

    tokenized_query = query.split()
    tokenized_query.extend(keywords)
    scores = bm25.get_scores(tokenized_query)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    result = {}
    for doc_index, score in sorted_scores:
        pmid = df.iloc[doc_index]['chunk_text']
        result[pmid] = score
    return list(result.keys())


def extract_keywords(text):
    documents = [text]
    tfidf = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_matrix = tfidf.fit_transform(documents)
    feature_names = tfidf.get_feature_names_out()
    tfidf_scores = zip(feature_names, tfidf_matrix.toarray()[0])
    sorted_words = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
    top_keywords = [word for word, score in sorted_words[:len(text) // 3 + 1]]

    unique_keywords = set()
    for phrase in top_keywords:
        words = phrase.split()
        unique_keywords.update(words)

    return list(unique_keywords)


def query_pre_process(query):
    proper_nouns = extract_keywords(query)
    if not proper_nouns:
        return []
    proper_nouns = [word for word in proper_nouns if word.lower() not in ENGLISH_STOP_WORDS] * 3
    return proper_nouns


def weightBM25(query, top_k=1):
    df = pd.read_csv('retrieval/QA_pair.csv')
    df['combined'] = str(df['PMID']) + " " + str(df['Question']) + " " + str(df['chunk_text'])

    texts = df['combined'].tolist()
    tokenized_texts = [doc.split() for doc in texts]
    bm25 = BM25(tokenized_texts)
    keywords = query_pre_process(query)
    result_pmids_scores = search(bm25, df, query, keywords, top_k=top_k)
    return result_pmids_scores
