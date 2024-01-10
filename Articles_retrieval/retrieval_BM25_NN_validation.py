import math
import random
from typing import List

import torch
from joblib import load
import pandas as pd
from torch import nn
from tqdm import tqdm


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

    def get_scores(self, query):
        scores = []
        for index, document in enumerate(self.corpus):
            score = self._score(document, query)
            scores.append((index, score))
        return scores

    def doc_length(self, document):
        return len(document)

    def common_terms(self, document, query):
        common = set(document) & set(query)
        return len(common), len(common) / len(document) if document else 0


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out



model_filename = 'neural_network_model_nn.pth'
model = NeuralNet(input_size=6, hidden_size=64, num_classes=1)
model.load_state_dict(torch.load(model_filename))
model.eval()


file_path = './PubmedDataSet.csv'
df = pd.read_csv(file_path)
df['combined'] = df['Abstract'].fillna("") + " " + df['PubDate'].fillna("") + " " + df['Authors'].fillna("")
texts = df['combined'].tolist()
tokenized_texts = [doc.split() for doc in texts]


bm25 = BM25(tokenized_texts)


def query_df_idf(query, bm25):
    df_values = [bm25.df.get(word, 0) for word in query if word in bm25.df]
    idf_values = [bm25.idf.get(word, 0) for word in query if word in bm25.idf]

    avg_df = sum(df_values) / len(df_values) if df_values else 0
    avg_idf = sum(idf_values) / len(idf_values) if idf_values else 0

    return avg_df, avg_idf


def get_top_abstracts(query, bm25, model, top_n=10):
    tokenized_query = query.split()
    avg_df, avg_idf = query_df_idf(tokenized_query, bm25)
    query_features = []
    pmids = df['PMID'].tolist()

    for doc_index, document in enumerate(bm25.corpus):
        bm25_score = bm25._score(document, tokenized_query)
        doc_len = bm25.doc_length(document)
        common_terms_count, common_terms_ratio = bm25.common_terms(document, tokenized_query)
        query_features.append([bm25_score, doc_len, common_terms_count, common_terms_ratio, avg_df, avg_idf])


    query_features_tensor = torch.tensor(query_features, dtype=torch.float32)


    with torch.no_grad():
        outputs = model(query_features_tensor).squeeze(1)

    _, top_indices = torch.topk(outputs, top_n)
    top_pmids = [pmids[i] for i in top_indices]

    return top_pmids




file_path = 'qap.csv'

df_qp_pair = pd.read_csv(file_path)

df_qp_pair = df_qp_pair.drop('Answer', axis=1)
data_list = df_qp_pair.values.tolist()

random.shuffle(data_list)
data_list = data_list[:200]


find_query_number = 0


for query, correct_pmid in tqdm(data_list, desc="Processing queries"):
    top_pmids = get_top_abstracts(query, bm25, model)
    if correct_pmid in top_pmids:
        find_query_number += 1

accuracy = find_query_number / len(data_list)
print(f"Accuracy: {accuracy * 100}%")
