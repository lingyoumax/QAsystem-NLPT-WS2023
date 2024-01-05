from opensearchpy import OpenSearch
import pandas as pd
import json

'''
    send all abstracts into opensearch
'''

csv_file_path = 'PubMed_dataset.csv'

df = pd.read_csv(csv_file_path)

host = 'opensearch'
port = 9200
auth = ('admin', 'admin')

opensearch_client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=False
)

index_name = 'articles'

for index, row in df.iterrows():
    print(index)
    document = row.to_dict()
    response = opensearch_client.index(index=index_name, body=json.dumps(document))
