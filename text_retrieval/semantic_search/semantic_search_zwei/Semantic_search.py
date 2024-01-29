import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from opensearchpy import OpenSearch
from tqdm import tqdm

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_embedding(text: str) -> torch.Tensor:
    """
    Convert text to a BERT embedding vector.

    :param text: Input text.
    :return: BERT embedding vector.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().to('cpu')


# Configure OpenSearch connection
host = 'opensearch'  # Replace with your OpenSearch host name or IP address
port = 9200  # Default HTTPS port is 9200
auth = ('admin', 'admin')  # Provide the correct username and password if your OpenSearch is configured with them

# Assume you already have an OpenSearch client
client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_auth=auth,
    use_ssl=True,  # Enable SSL connection
    verify_certs=False  # Validate SSL certificates
)

# mapping = {
#     "mappings": {
#         "properties": {
#             "embedding": {
#                 "type": "knn_vector",
#                 "dimension": 768
#             },
#             "pmid": {
#                 "type": "keyword"
#             }   
#         }
#     }
# }

# client.indices.create(index="articles_vectors", body=mapping)

# # Compute and store embeddings for all abstracts
# df = pd.read_csv('PubMed_dataset.csv')
# df['combined'] = df['Abstract'].fillna("") + " " + df['PubDate'].fillna("") + " " + df['Authors'].fillna("")
# for _, row in tqdm(df.iterrows(), total=df.shape[0]):
#     embedding = get_embedding(row['combined']).numpy().tolist()
#     document = {
#         'pmid': row['PMID'],
#         'embedding': embedding
#     }
#     client.index(index='articles_vectors', body=document)
    
# Example query
query = "How does the executive functioning of juvenile violent offenders, as assessed by tests like the Intra/Extradimensional Shift Test, Stockings of Cambridge Test, and Spatial Working Memory Test from the Cambridge Automated Neuropsychological Testing Battery, differ from that of non-violent offenders and normal controls, considering factors such as estimated Intelligence Quotient scores, experience of childhood trauma, and particularly the impact of childhood trauma on spatial working memory deficits?"
query_embedding = get_embedding(query).numpy().tolist()
#print(len(query_embedding))
# Perform vector search in OpenSearch
# Note: The query syntax below may need adjustment based on your OpenSearch version and setup
search_body = {
  "query": {
    "script_score": {
      "query": {
        "match_all": {}
      },
      "script": {
        "lang": "knn",
        "source": "knn_score",
        "params": {
          "field": "embedding",
          "query_value": query_embedding,
          "space_type": "l2"
        }
      }
    }
  }
}

response = client.search(index='articles_vectors', body=search_body)
print(response)
