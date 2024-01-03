from opensearchpy import OpenSearch
'''
    use opensearch to find relative abstracts
'''

opensearch_client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],
    http_auth=('admin', 'admin'),
    use_ssl=True,
    verify_certs=False
)

index_name = 'articles'


query = {
    "query": {
        "match": {
            "Abstract": "How can modern artificial intelligence-based techniques revolutionize healthcare by enhancing diagnostic accuracy in dental clinics and improving hearing in noisy environments, as demonstrated by the development of a model for detecting dental anomalies using panoramic dental images and the creation of 'smart' hearing aids and cochlear implants?"
        }
    }
}

response = opensearch_client.search(index=index_name, body=query)

for hit in response['hits']['hits']:
    print(f"ID: {hit['_id']}, Score: {hit['_score']}")
    print(hit['_source']['PMID'])
