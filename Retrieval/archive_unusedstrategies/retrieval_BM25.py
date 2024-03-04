import math
import random
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


df = pd.read_csv('./PubmedDataSet.csv')
df['combined'] = df['Abstract'].fillna("") + " " + df['PubDate'].fillna("") + " " + df['Authors'].fillna("")
texts = df['combined'].tolist()
tokenized_texts = [doc.split() for doc in texts]


bm25 = BM25(tokenized_texts)


def search(query: str, keywords):
    tokenized_query = query.split()
    tokenized_query.extend(keywords)
    scores = bm25.get_scores(tokenized_query)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:20]
    result = {}
    for doc_index, score in sorted_scores:
        pmid = df.iloc[doc_index]['PMID']
        result[pmid] = score
    return list(result.keys())


def search_org(query: str):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
    result = {}
    for doc_index, score in sorted_scores:
        pmid = df.iloc[doc_index]['PMID']
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
    years = re.findall(r'\b\d{4}\b', query)
    year_range = None
    if years:
        year_range = list(set([min(years), max(years)]))

    proper_nouns = extract_keywords(query)
    if not proper_nouns:
        if year_range:
            return year_range
        return []

    if year_range and len(year_range) == 1:
        for item in year_range * 5:
            if item:
                proper_nouns.append(item)

    proper_nouns = [word for word in proper_nouns if word.lower() not in ENGLISH_STOP_WORDS] * 3
    return proper_nouns


# 33543413
# query = "From abstract in 2022, in the context of cancer classification, what did the proposed hybrid approach consisting of ANFIS, FCM, and SA algorithm demonstrate?"
# 37705466
# query = 'Is Artificial Intelligence technology useful in gamete and embryo assessment and selection?'
# 38104897
# query = 'How has the emergence of deep tech startups influenced innovation in different sectors'
# keywords = query_pre_process(query)
# result_pmids_scores = search(query, keywords)
# print(result_pmids_scores)

file_path = 'qap.csv'
df_qp_pair = pd.read_csv(file_path)
df_qp_pair = df_qp_pair.drop('Answer', axis=1)
data_list = df_qp_pair.values.tolist()
random.shuffle(data_list)
data_list = data_list[:20]
find_query_number_weight = 0
find_query_number_org = 0

for query, correct_pmid in tqdm(data_list, desc="Processing queries"):
    keywords = query_pre_process(query)
    top_pmids = search(query, keywords)
    if correct_pmid in top_pmids:
        find_query_number_weight += 1
    top_pmids = search_org(query)
    if correct_pmid in top_pmids:
        find_query_number_org += 1

accuracy = find_query_number_org / len(data_list)
# print(f"Accuracy org: {accuracy * 100}%")
accuracy = find_query_number_weight / len(data_list)
print(f"Accuracy weight: {accuracy * 100}%")
