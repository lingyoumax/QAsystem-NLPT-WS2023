import pandas as pd
from rank_bm25 import BM25Okapi
from typing import List

'''
    Use BM25 to find relative abstracts
'''
# Read data from the CSV file
df = pd.read_csv('./updated_PubMed_dataset.csv')

# Combine fields to prepare documents
df['combined'] = df['Abstract'].fillna("") + " " + df['PubDate'].fillna("") + " " + df['Authors'].fillna("")
texts = df['combined'].tolist()
tokenized_texts = [doc.split(" ") for doc in texts]

# Build the BM25 model
bm25 = BM25Okapi(tokenized_texts)


def search(query: str) -> List[int]:
    """
    Search for documents most relevant to the query and return their PMIDs.

    :param query: A string representing the user's query.
    :return: A list of PMIDs for the documents most relevant to the query.
    """
    tokenized_query = query.split(" ")
    top_docs = bm25.get_top_n(tokenized_query, texts, n=5)
    # Use the 'combined' field to locate documents
    return [df[df['combined'] == doc]['PMID'].iloc[0] for doc in top_docs]


# Example usage
query = "What are the findings of the study regarding the impact of experience on nursing performance for nurses with high and low situational abilities, and how can training in EI potentially improve nursing performance?"
result_pmids = search(query)
print(result_pmids)
