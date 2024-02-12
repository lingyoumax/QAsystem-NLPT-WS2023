from pinecone import Pinecone
from FlagEmbedding import FlagModel

model = FlagModel('bge_large_fin',
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=True)

pc = Pinecone(api_key="621f7574-8c97-4f46-8c5e-186dd099d33b")
index = pc.Index("bge-fin")


def search_arxiv_texts(query):
    query_vector = model.encode_queries([query])[0].tolist()

    response = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )

    arxiv_texts = [match['metadata']['arxiv_text'] for match in response['matches']]

    return arxiv_texts


# query = "What was the purpose of the US Food and Drug Administration-cosponsored forum on laser-based imaging?"
# top_arxiv_texts = search_arxiv_texts(query)
# print(top_arxiv_texts)
