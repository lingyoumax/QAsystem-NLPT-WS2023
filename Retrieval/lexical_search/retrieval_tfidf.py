import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from typing import List


df = pd.read_csv('./updated_PubMed_dataset.csv')

df['combined'] = df['Abstract'].fillna("") + " " + df['PubDate'].fillna("") + " " + df['Authors'].fillna("")
texts = df['combined'].tolist()

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)


def search(query: str) -> List[int]:
    """
    Search for documents most relevant to the query and return their PMIDs.

    :param query: A string representing the user's query.
    :return: A list of PMIDs for the documents most relevant to the query.
    """
    query_vector = vectorizer.transform([query])
    cosine_similarities = linear_kernel(query_vector, tfidf_matrix).flatten()
    top_doc_indices = cosine_similarities.argsort()[-5:][::-1]  # 获取最相关的五个文档的索引
    return df.iloc[top_doc_indices]['PMID'].tolist()


query = "How can modern artificial intelligence-based techniques revolutionize healthcare by enhancing diagnostic accuracy in dental clinics and improving hearing in noisy environments, as demonstrated by the development of a model for detecting dental anomalies using panoramic dental images and the creation of 'smart' hearing aids and cochlear implants?"
result_pmids = search(query)
print(result_pmids)
