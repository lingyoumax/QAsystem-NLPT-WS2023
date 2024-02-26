import math
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import List, Dict
from joblib import dump
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.metrics import ndcg_score

class BM25:
    def __init__(self, corpus: List[List[str]], k1=1.5, b=0.75):
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

    def doc_length(self, document):
        return len(document)

    def common_terms(self, document, query):
        common = set(document) & set(query)
        return len(common), len(common) / len(document) if document else 0

import random

def query_df_idf(query, bm25):
    df_values = [bm25.df.get(word, 0) for word in query if word in bm25.df]
    idf_values = [bm25.idf.get(word, 0) for word in query if word in bm25.idf]

    avg_df = sum(df_values) / len(df_values) if df_values else 0
    avg_idf = sum(idf_values) / len(idf_values) if idf_values else 0

    return avg_df, avg_idf

num = 0

def prepare_features(data_list, bm25, num_negative_samples=2):
    features = []
    labels = []
    for query, pmid in data_list:
        tokenized_query = query.split()
        avg_df, avg_idf = query_df_idf(tokenized_query, bm25)

        try:
            row = df[df['PMID'] == pmid].iloc[0]

            for _ in range(num_negative_samples + 1):  # +1 is for including the positive sample

                if _ == 0:  # The first sample is the positive sample
                    row = df[df['PMID'] == pmid].iloc[0]
                else:  # Generate negative samples
                    row = df.sample().iloc[0]
                doc_index = row.name
                document = bm25.corpus[doc_index]
                bm25_score = bm25._score(document, tokenized_query)
                doc_len = bm25.doc_length(document)
                common_terms_count, common_terms_ratio = bm25.common_terms(document, tokenized_query)

                features.append([bm25_score, doc_len, common_terms_count, common_terms_ratio, avg_df, avg_idf])
                label = 1 if _ == 0 else 0
                labels.append(label)
                global num
                num += 1
        except:
            continue
    return features, labels

file_path = './PubmedDataSet.csv'
df = pd.read_csv(file_path)
df['combined'] = df['Abstract'].fillna("") + " " + df['PubDate'].fillna("") + " " + df['Authors'].fillna("")
texts = df['combined'].tolist()
tokenized_texts = [doc.split() for doc in texts]

bm25 = BM25(tokenized_texts)

file_path_qap = 'qap.csv'
df_qp_pair = pd.read_csv(file_path_qap)
df_qp_pair = df_qp_pair.drop('Answer', axis=1)
data_list = df_qp_pair.values.tolist()

features, labels = prepare_features(data_list, bm25)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

print('Fitting LambdaMART model...')
lambdamart_model = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    learning_rate=0.05,
    n_estimators=1000,
)

samples_per_query = 3
num_queries = len(X_train) // samples_per_query
query_lens = [samples_per_query] * num_queries
lambdamart_model.fit(X_train, y_train, group=query_lens)

model_filename = 'lambdamart_model.joblib'
dump(lambdamart_model, model_filename)
print(f"Model saved to {model_filename}")

predictions = lambdamart_model.predict(X_test)
print("NDCG Score:", ndcg_score([y_test], [predictions]))
