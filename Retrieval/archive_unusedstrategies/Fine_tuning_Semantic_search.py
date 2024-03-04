import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_embedding(text_list: list) -> torch.Tensor:
    """
    Convert a list of texts to BERT embeddings vectors.

    :param text_list: List of input texts.
    :return: Tensor of BERT embeddings.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()


# Load your datasets
training_df = pd.read_csv('./qap.csv')
abstracts_df = pd.read_csv('./PubmedDataSet.csv')

# Merge datasets on PMID
merged_df = training_df.merge(abstracts_df, how='left', left_on='PMID', right_on='PMID')


# Define a PyTorch dataset
class SemanticSearchDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        question = str(self.dataframe.iloc[idx]['Question'])
        abstract = str(self.dataframe.iloc[idx]['Abstract'])

        if not question.strip():
            question = "empty"
        if not abstract.strip():
            abstract = "empty"

        return {
            'question': question,
            'abstract': abstract
        }


# Create dataset and dataloader
dataset = SemanticSearchDataset(merged_df, tokenizer)
print('dataset finished')
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Fine-tuning the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)

from torch.nn.functional import cosine_similarity
import torch.nn as nn


# Define a simple loss function based on cosine similarity
class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, embeddings1, embeddings2):
        # Cosine similarity ranges from -1 to 1. Higher is better.
        cosine_loss = 1 - cosine_similarity(embeddings1, embeddings2)
        return cosine_loss.mean()


# Initialize the loss function
loss_fn = CosineSimilarityLoss()

for epoch in range(3):  # Number of training epochs
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader):
        # Reset gradients
        optimizer.zero_grad()

        # Compute embeddings
        question_embeddings = get_embedding(batch['question']).to(device)
        abstract_embeddings = get_embedding(batch['abstract']).to(device)

        # Calculate cosine similarity loss
        loss = loss_fn(question_embeddings, abstract_embeddings)
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(data_loader)}")

# Save the fine-tuned model
model.save_pretrained('./fine_tuned')
