from pinecone import Pinecone
from FlagEmbedding import FlagModel

model = FlagModel('./retrieval/bge_large_fin',
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=True)

print("Model loaded successfully")

pc = Pinecone(api_key="621f7574-8c97-4f46-8c5e-186dd099d33b")
index = pc.Index("bge-fin")


def generate_year_range(start_year, end_year):
    """Generate a list of years as strings from start_year to end_year."""
    return [str(year) for year in range(int(start_year), int(end_year) + 1)]


def search_arxiv_texts(query, year, authors=None):
    query_vector = model.encode_queries([query])[0].tolist()

    # Initialize filters
    filters = {}
    if year:
        start_year = year[0]
        end_year = year[1]
        years = generate_year_range(start_year, end_year)
        filters['publishedDate'] = {
            "$in": years
        }
    if authors:
        if not isinstance(authors, list):
            authors = [authors]
        authors = [author.lower() for author in authors]
        filters['authors'] = {
            "$in": authors
        }

    response = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
        filter=filters  # Apply filters to the query, including the author filter
    )

    arxiv_texts = [match['metadata']['arxiv_text'] for match in response['matches']]

    return arxiv_texts