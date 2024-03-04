import math
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import List, Dict

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

def query_df_idf(query, bm25):
    df_values = [bm25.df.get(word, 0) for word in query if word in bm25.df]
    idf_values = [bm25.idf.get(word, 0) for word in query if word in bm25.idf]

    avg_df = sum(df_values) / len(df_values) if df_values else 0
    avg_idf = sum(idf_values) / len(idf_values) if idf_values else 0

    return avg_df, avg_idf

def prepare_features(data_list, bm25, num_negative_samples=1):
    features = []
    labels = []
    for query, pmid in data_list:
        tokenized_query = query.split()
        avg_df, avg_idf = query_df_idf(tokenized_query, bm25)
        for _ in range(num_negative_samples + 1):  # +1 is to include the positive sample
            if _ == 0:  # The first sample is the positive one
                try:
                    row = df[df['PMID'] == pmid].iloc[0]
                except IndexError:
                    continue
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

    return features, labels

# Read data and preprocess
file_path = './PubmedDataSet.csv'
df = pd.read_csv(file_path)
df['combined'] = df['Abstract'].fillna("") + " " + df['PubDate'].fillna("") + " " + df['Authors'].fillna("")
texts = df['combined'].tolist()
tokenized_texts = [doc.split() for doc in texts]

# Initialize BM25 object
bm25 = BM25(tokenized_texts)

# Prepare training data
file_path_qap = 'qap.csv'
df_qp_pair = pd.read_csv(file_path_qap)
df_qp_pair = df_qp_pair.drop('Answer', axis=1)
data_list = df_qp_pair.values.tolist()

# Prepare features and labels
features, labels = prepare_features(data_list, bm25)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)
print('fitting')

# Train Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Save the model to a file
from joblib import dump
model_filename = 'logistic_regression_model_6features.joblib'
dump(lr, model_filename)
print(f"Model saved to {model_filename}")

# Predict and evaluate
predictions = lr.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
