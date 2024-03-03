import pandas as pd
from mongodb import question_collection
from retrieval.bm25 import weightBM25, BM25
from retrieval.semantic_search import search_arxiv_texts
from Answer_generation.SFT_inference import inference

df = pd.read_csv('./retrieval/PubmedDataSet.csv')
texts = df['Abstract'].tolist()
tokenized_texts = [doc.split() for doc in texts]
bm25_abstract = BM25(tokenized_texts)
df2 = pd.read_csv('./retrieval/splitted_pubmed_data_NLTK.csv')


def retrieval(question, year, author):
    res_semantic = search_arxiv_texts(question, year, author)
    if not res_semantic:
        res_semantic = ""
    else:
        res_semantic = res_semantic[0]
    res_lex = weightBM25(question, df, df2, year, author)[1]
    res_semantic = search_arxiv_texts(question + ' ' + res_semantic, year, author)
    if not res_semantic:
        res_semantic = ""
    else:
        res_semantic = res_semantic[0]
    res_lex = weightBM25(question + ' ' + res_lex, df, df2, year, author)[1]
    res_lex = res_lex.replace('\n', '')


    if res_semantic == res_lex or res_lex in res_semantic:
        return res_lex
    else:
        return res_lex + " " + res_semantic

# What are some challenges and limitations in using data mining techniques in healthcare?

def answerGeneration(question, reference):
    answer = inference(instructions=[{"instruction": reference, "input": question}],
        batch_size=1, temperature=0.7, top_k=1,
        top_p=0.2)
    print(answer)
    return answer[0]['generated']


async def process(TIME_STAMP, question, year, author):
    query = {"time_stamp": TIME_STAMP}

    # Choose the retrieval method based on type
    reference = retrieval(question, year, author)
    reference = reference.replace("\n", " ")
    # Update the question in the database with retrieval set to true and store the reference
    update = {"$set": {"retrieval": True, 'reference': reference}}
    question_collection.find_one_and_update(query, update)

    # Choose the answer generation model based on type
    answer = answerGeneration(question, reference)
    # Update the question in the database with answerGeneration set to true, write the answer and set output to true
    update = {"$set": {'answerGeneration': True, 'answer': answer, 'output': True}}
    question_collection.find_one_and_update(query, update)
