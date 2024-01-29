import numpy as np
from sentence_transformers import SentenceTransformer, util


def process(question, sentences, top_n=3):
    # 加载预训练的模型
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 将问题和句子转换为嵌入
    question_embedding = model.encode(question, convert_to_tensor=True)
    sentences_embeddings = model.encode(sentences, convert_to_tensor=True)

    # 计算问题和每个句子之间的余弦相似度
    cosine_scores = util.pytorch_cos_sim(question_embedding, sentences_embeddings)

    # 获取相似度分数，并将其转换为一维数组
    scores = cosine_scores[0].cpu().numpy()

    # 获取相似度最高的前三个句子的索引
    top_scores_indices = np.argsort(-scores)[:top_n]

    # 返回相似度最高的前三个句子
    top_sentences = [sentences[index] for index in top_scores_indices]

    return top_sentences


import pandas as pd

# 读取CSV文件
df = pd.read_csv('qad.csv')  # 从第二行开始读取，第一行被跳过
# df = df.drop(columns=['Answer'])

for index, row in df.iterrows():
    question = row['Question']
    chunk_text = row['chunk_text'].split('.')
    top_n = process(question, chunk_text)
    print(question)
    print(chunk_text)
    print(top_n)
    break


