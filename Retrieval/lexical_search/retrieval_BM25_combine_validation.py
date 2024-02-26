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

def get_top_abstracts_NN(query, bm25, model, top_n=20):
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

def get_top_abstracts_LR(query, bm25, model, top_n=20):
    tokenized_query = query.split()
    avg_df, avg_idf = query_df_idf(tokenized_query, bm25)
    query_features = []
    pmids = df['PMID'].tolist()

    for doc_index, document in enumerate(bm25.corpus):
        bm25_score = bm25._score(document, tokenized_query)
        doc_len = bm25.doc_length(document)
        common_terms_count, common_terms_ratio = bm25.common_terms(document, tokenized_query)
        query_features.append(
            [bm25_score, doc_len, common_terms_count, common_terms_ratio, avg_df, avg_idf])

    predictions = model.predict_proba(query_features)[:, 1]
    top_indices = sorted(range(len(predictions)), key=lambda i: predictions[i], reverse=True)[:top_n]
    top_pmids = [pmids[i] for i in top_indices]

    return top_pmids

def search(query: str) -> List[tuple]:
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:20]
    result = {}
    for doc_index, score in sorted_scores:
        pmid = df.iloc[doc_index]['PMID']
        result[pmid] = score
    return list(result.keys())

def get_top_abstracts_MART(query, bm25, model, top_n=10):
    tokenized_query = query.split()
    avg_df, avg_idf = query_df_idf(tokenized_query, bm25)
    query_features = []
    pmids = df['PMID'].tolist()

    for doc_index, document in enumerate(bm25.corpus):
        bm25_score = bm25._score(document, tokenized_query)
        doc_len = bm25.doc_length(document)
        common_terms_count, common_terms_ratio = bm25.common_terms(document, tokenized_query)
        row = df.iloc[doc_index]
        query_features.append([bm25_score, doc_len, common_terms_count, common_terms_ratio, avg_df, avg_idf])

    predictions = model.predict(query_features)
    top_indices = sorted(range(len(predictions)), key=lambda i: predictions[i], reverse=True)[:top_n]
    top_pmids = [pmids[i] for i in top_indices]

    return top_pmids

file_path = 'qap.csv'
df_qp_pair = pd.read_csv(file_path)
df_qp_pair = df_qp_pair.drop('Answer', axis=1)
data_list = df_qp_pair.values.tolist()
random.shuffle(data_list)
data_list = data_list[:150]
model_filename = 'logistic_regression_model_6features.joblib'
lr_model = load(model_filename)
model_filename = 'lambdamart_model.joblib'
lambdamart_model = load(model_filename)

find_query_number0 = 0
find_query_number2 = 0
find_query_number3 = 0
find_query_number4 = 0
find_query_number5 = 0
find_query_number6 = 0
find_query_number7 = 0
find_query_number8 = 0

for query, correct_pmid in tqdm(data_list, desc="Processing queries"):
    top_pmids_MART = get_top_abstracts_MART(query, bm25, lambdamart_model)
    top_pmids_NN = get_top_abstracts_NN(query, bm25, model)
    top_pmids_LR = get_top_abstracts_LR(query, bm25, lr_model)
    search_result = search(query)
    top_pmids_intersection = list(set(top_pmids_LR).intersection(search_result).intersection(top_pmids_NN))
    if len(top_pmids_intersection) < 5:
        top_pmids_intersection = search_result[:10]
    if correct_pmid in top_pmids_intersection:
        find_query_number0 += 1
    if correct_pmid in search_result[:10]:
        find_query_number2 += 1
    if correct_pmid in top_pmids_NN[:10]:
        find_query_number3 += 1
    if correct_pmid in top_pmids_LR[:10]:
        find_query_number4 += 1
    if correct_pmid in top_pmids_MART[:10]:
        find_query_number8 += 1

    top_pmids_intersection = list(set(top_pmids_LR).intersection(search_result))
    if len(top_pmids_intersection) < 5:
        top_pmids_intersection = search_result[:10]
    if correct_pmid in top_pmids_intersection:
        find_query_number5 += 1

    top_pmids_intersection = list(set(top_pmids_LR).intersection(top_pmids_NN))
    if len(top_pmids_intersection) < 5:
        top_pmids_intersection = search_result[:10]
    if correct_pmid in top_pmids_intersection:
        find_query_number6 += 1

    top_pmids_intersection = list(set(search_result).intersection(top_pmids_NN))
    if len(top_pmids_intersection) < 5:
        top_pmids_intersection = search_result[:10]
    if correct_pmid in top_pmids_intersection:
        find_query_number7 += 1

accuracy = find_query_number8 / len(data_list)
print(f"Accuracy MART: {accuracy * 100}%")
accuracy = find_query_number7 / len(data_list)
print(f"Accuracy combine NN org: {accuracy * 100}%")
accuracy = find_query_number6 / len(data_list)
print(f"Accuracy combine NN LR: {accuracy * 100}%")
accuracy = find_query_number5 / len(data_list)
print(f"Accuracy combine LR org: {accuracy * 100}%")
accuracy = find_query_number0 / len(data_list)
print(f"Accuracy combine NN LR org: {accuracy * 100}%")
accuracy2 = find_query_number2 / len(data_list)
print(f"Accuracy org: {accuracy2 * 100}%")
accuracy3 = find_query_number3 / len(data_list)
print(f"Accuracy NN: {accuracy3 * 100}%")
accuracy4 = find_query_number4 / len(data_list)
print(f"Accuracy LR: {accuracy4 * 100}%")
