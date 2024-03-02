import math
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from typing import List
import yake

nlp = spacy.load("en_core_web_sm")
kw_extractor = yake.KeywordExtractor(n=1,
                                     dedupLim=0.9,
                                     top=10,
                                     features=None)


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


def search(query: str, df, keywords, bm25, top_k):
    tokenized_query = query.split()
    tokenized_query.extend(keywords)
    scores = bm25.get_scores(tokenized_query)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    result = {}
    if top_k == 1:
        for doc_index, score in sorted_scores:
            return [df.iloc[doc_index]['PMID'], df.iloc[doc_index]['chunk_text']]

    for doc_index, score in sorted_scores:
        pmid = df.iloc[doc_index]['PMID']
        result[pmid] = score
    return list(result.keys())


def extract_keywords(text):
    keywords = kw_extractor.extract_keywords(text)
    keywords_set = {word for word, _ in keywords}
    return list(set(keywords_set))


def query_pre_process(query):
    proper_nouns = extract_keywords(query)
    proper_nouns = [word for word in proper_nouns if word.lower() not in ENGLISH_STOP_WORDS] * 1
    return proper_nouns





def weightBM25(query, df, df2, year=None, author=None):
    if year:
        [year0, year1] = year
        filtered_df = df[df['PubDate'] != 'unknown']
        df_filtered = filtered_df[(filtered_df['PubDate'].astype(int) >= int(year0)) & (filtered_df['PubDate'].astype(int) <= int(year1))]
    else:
        df_filtered = df

    if author:
        df_filtered = df_filtered[df_filtered['Authors'].str.contains(author, case=False, na=False)]



    texts = df_filtered['Abstract'].tolist()
    tokenized_texts = [doc.split() for doc in texts]
    bm25_abstract = BM25(tokenized_texts)

    keywords = query_pre_process(query)
    result_pmids_scores = search(query, df_filtered, keywords, bm25_abstract, top_k=30)

    mask = df2['PMID'].isin([pmid for pmid in result_pmids_scores])
    df_t = df2[mask]

    texts = df_t['chunk_text'].tolist()
    tokenized_texts = [doc.split() for doc in texts]
    bm25_chunk = BM25(tokenized_texts)

    result_chunk = search(query, df_t, keywords, bm25_chunk, top_k=1)


    return result_chunk

