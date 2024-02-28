from pinecone import Pinecone
from FlagEmbedding import FlagModel


def search_arxiv_texts(query):
    model = FlagModel('/Users/liuziwei/Desktop/NLP/QAsystem-NLPT-WS2023/backend/retrieval/bge_large_fin',
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                      use_fp16=True)

    pc = Pinecone(api_key="621f7574-8c97-4f46-8c5e-186dd099d33b")
    Index = pc.Index("bge-fin")

    query_vector = model.encode_queries([query])[0].tolist()

    response = Index.query(
        vector=query_vector,
        top_k=1,
        include_metadata=True
    )

    arxiv_texts = [str(match['metadata']['arxiv_text']) for match in response['matches']]
    pmid = [int(match['metadata']['pmid']) for match in response['matches']]
    # ans = {}
    # for i in range(5):
    #     ans[pmid[i]] = arxiv_texts[i]

    return [pmid[0], arxiv_texts[0]]


