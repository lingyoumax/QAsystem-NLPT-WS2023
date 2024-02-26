from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import math
import pandas as pd
from joblib import load
import torch
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import List, Dict
from joblib import dump

from sklearn.svm import SVC
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

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

def bert_vectorize(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu().tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

def get_top_abstracts_bert(query, bm25, model, df, doc_vectors, top_n=10):
    query_vector = bert_vectorize(query, tokenizer, bert_model, device)
    pmids = df['PMID'].tolist()
    query_features = []
    for doc_index, document in enumerate(bm25.corpus):
        doc_vector = doc_vectors[doc_index]
        combined_features = query_vector + doc_vector
        query_features.append(combined_features)
    predictions = model.predict_proba(query_features)[:, 1]
    top_indices = sorted(range(len(predictions)), key=lambda i: predictions[i], reverse=True)[:top_n]
    top_pmids = [pmids[i] for i in top_indices]
    return top_pmids

model_filename = 'svm_model_bert_features.joblib'
svm_model = load(model_filename)

file_path = './PubmedDataSet.csv'
df = pd.read_csv(file_path)
df['combined'] = df['Abstract'].fillna("") + " " + df['PubDate'].fillna("") + " " + df['Authors'].fillna("")
texts = df['combined'].tolist()
tokenized_texts = [doc.split() for doc in texts]
bm25 = BM25(tokenized_texts)

def save_doc_vectors(df, tokenizer, model, device, filename='doc_vectors.pt'):
    doc_vectors = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Vectorizing documents"):
        doc_vector = bert_vectorize(row['combined'], tokenizer, model, device)
        doc_vectors.append(doc_vector)
    torch.save(doc_vectors, filename)

save_doc_vectors(df, tokenizer, bert_model, device)
doc_vectors = torch.load('doc_vectors.pt')

file_path_qap = 'qap.csv'
df_qp_pair = pd.read_csv(file_path_qap)
df_qp_pair = df_qp_pair.drop('Answer', axis=1)
data_list = df_qp_pair.values.tolist()
random.seed(42)
random.shuffle(data_list)
data_list = data_list[:100]

find_query_number = 0
for query, correct_pmid in tqdm(data_list, desc="Processing queries"):
    top_pmids = get_top_abstracts_bert(query, bm25, svm_model, df, doc_vectors)
    if correct_pmid in top_pmids:
        find_query_number += 1

accuracy = find_query_number / len(data_list)
print(f"Accuracy: {accuracy * 100}%")
