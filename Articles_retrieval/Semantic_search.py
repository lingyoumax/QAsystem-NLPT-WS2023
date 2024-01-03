import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from opensearchpy import OpenSearch

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_embedding(text: str) -> torch.Tensor:
    """
    Convert text to a BERT embedding vector.

    :param text: Input text.
    :return: BERT embedding vector.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


# Configure OpenSearch connection
host = 'localhost'  # Replace with your OpenSearch host name or IP address
port = 9200  # Default HTTPS port is 9200
auth = ('admin', 'admin')  # Provide the correct username and password if your OpenSearch is configured with them

# Assume you already have an OpenSearch client
client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_auth=auth,
    use_ssl=True,  # Enable SSL connection
    verify_certs=False  # Validate SSL certificates
)

# Compute and store embeddings for all abstracts
df = pd.read_csv('./updated_PubMed_dataset.csv')
df['combined'] = df['Abstract'].fillna("") + " " + df['PubDate'].fillna("") + " " + df['Authors'].fillna("")
for _, row in df.iterrows():
    embedding = get_embedding(row['combined']).numpy().tolist()
    document = {
        'pmid': row['PMID'],
        'embedding': embedding
    }
    client.index(index='articles_vectors', body=document)

# Example query
query = "Your query here"
query_embedding = get_embedding(query).numpy().tolist()

# Perform vector search in OpenSearch
# Note: The query syntax below may need adjustment based on your OpenSearch version and setup
search_body = {
    "query": {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": query_embedding}
            }
        }
    }
}
response = client.search(index='articles_vectors', body=search_body)
print(response)
