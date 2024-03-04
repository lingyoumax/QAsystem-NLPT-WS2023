import numpy as np
from sentence_transformers import SentenceTransformer, util


def process(question, sentences, top_n=3):
    
    model = SentenceTransformer('all-MiniLM-L6-v2')

    question_embedding = model.encode(question, convert_to_tensor=True)
    sentences_embeddings = model.encode(sentences, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(question_embedding, sentences_embeddings)

    scores = cosine_scores[0].cpu().numpy()

    top_scores_indices = np.argsort(-scores)[:top_n]

    top_sentences = [sentences[index] for index in top_scores_indices]

    return top_sentences


import pandas as pd

df = pd.read_csv('qad.csv') 
# df = df.drop(columns=['Answer'])

for index, row in df.iterrows():
    question = row['Question']
    chunk_text = row['chunk_text'].split('.')
    top_n = process(question, chunk_text)
    print(question)
    print(chunk_text)
    print(top_n)
    break


