import os
import torch
from sklearn.cluster import AgglomerativeClustering
import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


'''
    use data from PubMed_dataset.csv to cluster abstracts discuss similar topic together,
    and use each group of similar abstract to generate QA-abstracts pairs. 
    For example, abstract A and abstract B discuss similar topic, and QA are based on both abstracts.
'''



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_version = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_version)
model = BertModel.from_pretrained(bert_version).to(device).eval()


# Function: Structuring abstract data into text form
def create_text_representation(pub_date, authors, abstract, keywords, title):
    text = f"Date: {pub_date} Authors: {' '.join(authors)} Title: {title} Keywords: {', '.join(keywords)} Abstract: {abstract}"
    return text


def encode_with_bert(texts, batch_size=8):
    all_encoded_vectors = []
    # Use tqdm to show progress
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", unit="batch"):
        batch_texts = texts[i:i + batch_size]
        encodings = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512).to(
            device)
        with torch.no_grad():
            outputs = model(**encodings)
        batch_encoded_vectors = outputs.last_hidden_state.mean(dim=1)
        all_encoded_vectors.append(batch_encoded_vectors.cpu())  # Move to CPU to save GPU memory
        torch.cuda.empty_cache()  # Clear unused memory
    return torch.cat(all_encoded_vectors, dim=0)


# Prepare abstract data and include PMID
def prepare_abstracts_data(csv_filename):
    abstracts_data = []
    pmids = []

    with open(csv_filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            date = row['PubDate'] if row['PubDate'] != 'unknown' else (
                row['BeginningDate'] if row['BeginningDate'] != 'unknown' else row['EndingDate'])
            keywords = row['Keywords'].split(', ')
            title = row['ArticleTitle']
            authors = row['Authors'].split(', ')
            pmids.append(row['PMID'])
            abstracts_data.append((date, authors, row['Abstract'], keywords, title))

    return abstracts_data, pmids
    # return pmids


# Load data
csv_filename = './PubMed_dataset.csv'
abstracts_data, pmids = prepare_abstracts_data(csv_filename)
print('Preparation finished')
# Text encoding
texts = [create_text_representation(*abstract) for abstract in abstracts_data]
encoded_vectors = encode_with_bert(texts)
print('Encoding finished')

# Move encoded vectors to CPU and convert to NumPy array
encoded_vectors_np = encoded_vectors.cpu().numpy()
# Save encoded vectors to local file
np.save('encoded_vectors.npy', encoded_vectors_np)

encoded_vectors_np = np.load('encoded_vectors.npy')

# Normalize vectors
normalized_vectors = normalize(encoded_vectors_np)
# Check if a pre-computed similarity matrix exists, else compute and save it
similarity_matrix_file = 'similarity_matrix.npy'
if os.path.exists(similarity_matrix_file):
    similarity_matrix = np.load(similarity_matrix_file)
else:
    similarity_matrix = cosine_similarity(normalized_vectors)
    np.save(similarity_matrix_file, similarity_matrix)

print('Cosine similarity calculation finished')


def split_matrix(matrix, pmids, max_size=30000):
    """Split the matrix into smaller matrices, and split pmids accordingly"""
    for i in range(0, len(matrix), max_size):
        submatrix = matrix[i:i + max_size, i:i + max_size]
        sub_pmids = pmids[i:i + max_size]  # Split pmids synchronously

        # Perform clustering and save results
        cluster_and_save(submatrix, sub_pmids, i)


i = 0


def cluster_and_save(similarity_matrix, sub_pmids, idx):
    """Apply clustering to submatrix and save the results to a file"""
    # Perform clustering using AgglomerativeClustering
    clustering = AgglomerativeClustering(n_clusters=2800, affinity='precomputed', linkage='complete')
    labels = clustering.fit_predict(1 - similarity_matrix)

    # Write PMIDs of each cluster into a file
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(sub_pmids[idx])

    # Write clustering results to a text file
    global i
    with open(f'clusters_pmids_{i}.txt', 'w') as f:
        for label, pmids in clusters.items():
            f.write(f"Cluster {label}: " + ', '.join(pmids) + '\n')

    print(f"Clustering results written to 'clusters_pmids_{i}.txt'")
    i += 1


# Split similarity_matrix and perform clustering
split_matrix(similarity_matrix, pmids)
