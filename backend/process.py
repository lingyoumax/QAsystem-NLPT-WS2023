import pandas as pd
from mongodb import question_collection
from retrieval.bm25 import weightBM25, BM25
from retrieval.semantic_search import search_arxiv_texts

df = pd.read_csv('./retrieval/PubmedDataSet.csv')
texts = df['Abstract'].tolist()
tokenized_texts = [doc.split() for doc in texts]
bm25_abstract = BM25(tokenized_texts)
df2 = pd.read_csv('./retrieval/splitted_pubmed_data_NLTK.csv')


def retrieval(question, year):
    res_semantic = search_arxiv_texts(question)[1]
    res_lex = weightBM25(question, df, df2, year)[1]
    question += ' ' + res_semantic + ' ' + res_lex

    res_semantic = search_arxiv_texts(question)[1]
    res_lex = weightBM25(question, df, df2, year)[1]

    if res_semantic == res_lex:
        return res_lex
    else:
        return res_lex + " " + res_semantic


async def answerGeneration(type, question, reference):
    return 1


async def process(TIME_STAMP, question, year):
    query = {"time_stamp": TIME_STAMP}

    # Choose the retrieval method based on type
    reference = retrieval(question, year)
    # Update the question in the database with retrieval set to true and store the reference
    update = {"$set": {"retrieval": True, 'reference': reference}}
    question_collection.find_one_and_update(query, update)

    print(reference)
    return

    # Choose the answer generation model based on type
    answer = await answerGeneration(type, question, reference)
    # Update the question in the database with answerGeneration set to true, write the answer and set output to true
    update = {"$set": {'answerGeneration': True, 'answer': answer, 'output': True}}
    question_collection.find_one_and_update(query, update)
