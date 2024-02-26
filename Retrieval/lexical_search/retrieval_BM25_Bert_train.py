import pickle

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertSimilarityNet(nn.Module):
    def __init__(self, bert_model_query, bert_model_doc):
        super(BertSimilarityNet, self).__init__()
        self.bert_query = bert_model_query
        self.bert_doc = bert_model_doc
        self.fc1 = nn.Linear(768 * 2, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, query, doc):
        # 解构query和doc字典
        query_input_ids = query['input_ids'].squeeze(1)
        query_attention_mask = query['attention_mask'].squeeze(1)
        doc_input_ids = doc['input_ids'].squeeze(1)
        doc_attention_mask = doc['attention_mask'].squeeze(1)

        query_vec = self.bert_query(query_input_ids, attention_mask=query_attention_mask).last_hidden_state.mean(1)
        doc_vec = self.bert_doc(doc_input_ids, attention_mask=doc_attention_mask).last_hidden_state.mean(1)

        combined = torch.cat((query_vec, doc_vec), dim=1)
        output = self.fc1(combined)
        output = self.fc2(output)
        output = self.fc3(output)
        return torch.sigmoid(output)


class QAPDataset(Dataset):
    def __init__(self, filename, tokenizer, df, negative_sample_ratio=1):
        self.tokenizer = tokenizer
        self.df = df
        self.data = []

        df_qp_pair = pd.read_csv(filename)
        df_qp_pair = df_qp_pair.drop('Answer', axis=1)

        for _, row in df_qp_pair.iterrows():
            try:
                positive_doc = df[df['PMID'] == row['PMID']].iloc[0]['combined']
            except:
                continue
            query = row['Question']
            positive_doc = df[df['PMID'] == row['PMID']].iloc[0]['combined']
            self.data.append((query, positive_doc, 1))  # Positive sample
            # Add negative samples
            for _ in range(negative_sample_ratio):
                negative_doc = df.sample().iloc[0]['combined']
                self.data.append((query, negative_doc, 0))  # Negative sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, doc, label = self.data[idx]
        # 确保在这里对query和doc应用padding和truncation
        query_enc = self.tokenizer(query, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        doc_enc = self.tokenizer(doc, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        return query_enc, doc_enc, torch.tensor(label, dtype=torch.float32)


from tqdm import tqdm


def train_model(model, train_loader, criterion, optimizer, epochs=3):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # 使用tqdm包装train_loader
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            query_enc, doc_enc, labels = batch

            # 由于query_enc和doc_enc是dict，需要对它们的每个值应用.to(device)
            query_enc = {k: v.to(device) for k, v in query_enc.items()}
            doc_enc = {k: v.to(device) for k, v in doc_enc.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(query_enc, doc_enc)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # 清理GPU缓存
            del query_enc, doc_enc, labels, outputs, loss
            torch.cuda.empty_cache()

        # 每个epoch结束后打印平均损失
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}')


# 初始化BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model_query = BertModel.from_pretrained('bert-base-uncased')
bert_model_doc = BertModel.from_pretrained('bert-base-uncased')

# 加载数据集
file_path_qap = 'qap.csv'
df = pd.read_csv('./PubmedDataSet.csv')
df['combined'] = df['Abstract'].fillna("") + " " + df['PubDate'].fillna("") + " " + df['Authors'].fillna("")
dataset = QAPDataset(file_path_qap, tokenizer, df)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
print('train data prepared')
# 初始化和训练模型
model = BertSimilarityNet(bert_model_query, bert_model_doc).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_model(model, train_loader, criterion, optimizer)
print('training finished')
torch.save(model.bert_query.state_dict(), 'bert_query_model.pth')
torch.save(model.bert_doc.state_dict(), 'bert_doc_model.pth')
torch.save(model.fc1.state_dict(), 'fc1_model.pth')
torch.save(model.fc2.state_dict(), 'fc2_model.pth')
torch.save(model.fc3.state_dict(), 'fc3_model.pth')


# 加载bert_doc模型
bert_model_doc = BertModel.from_pretrained('bert-base-uncased')
bert_model_doc.load_state_dict(torch.load('bert_doc_model.pth'))
bert_model_doc = bert_model_doc.to(device)
bert_model_doc.eval()


print('calculating doc_vec...')
# 处理df['combined']数据并保存向量
vectors = []
for _, row in df.iterrows():
    doc = row['combined']
    doc_enc = tokenizer(doc, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        doc_vec = bert_model_doc(**doc_enc).last_hidden_state.mean(1).squeeze().cpu().numpy()
    vectors.append((row['PMID'], doc_vec))

# 保存PMID和向量
with open('doc_vectors.pkl', 'wb') as f:
    pickle.dump(vectors, f)
print('doc_vectors.pkl saved')







import torch
from transformers import BertModel, BertTokenizer
import pickle

# 加载BERT模型
bert_query = BertModel.from_pretrained('bert-base-uncased')
bert_query.load_state_dict(torch.load('bert_query_model.pth'))
bert_query.eval()

# 加载全连接层
fc1 = nn.Linear(768 * 2, 512)
fc2 = nn.Linear(512, 64)
fc3 = nn.Linear(64, 1)
fc1.load_state_dict(torch.load('fc1_model.pth'))
fc2.load_state_dict(torch.load('fc2_model.pth'))
fc3.load_state_dict(torch.load('fc3_model.pth'))
# 加载文档向量
with open('doc_vectors.pkl', 'rb') as f:
    doc_vectors = pickle.load(f)

print('start validation')


def get_top_pmids(query, top_n=5):
    # 处理query
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    query_enc = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    query_vec = bert_query(**query_enc).last_hidden_state.mean(1)

    # 计算得分
    scores = []
    for pmid, doc_vec in doc_vectors:
        combined_vec = torch.cat((query_vec, torch.tensor(doc_vec).unsqueeze(0)), dim=1)
        output = fc1(combined_vec)
        output = fc2(output)
        output = fc3(output)
        scores.append((pmid, output.item()))

    # 排序并获取得分最高的top_n个PMID
    top_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    top_pmids = [score[0] for score in top_scores]

    return top_pmids


# 加载验证集
file_path = 'qap.csv'
df_qp_pair = pd.read_csv(file_path)
df_qp_pair = df_qp_pair.drop('Answer', axis=1)
data_list = df_qp_pair.values.tolist()
random.shuffle(data_list)
data_list = data_list[:100]

# 计算准确率
correct = 0
for query, correct_pmid in data_list:
    top_pmids = get_top_pmids(query)
    if correct_pmid in top_pmids:
        correct += 1

accuracy = correct / len(data_list)
print(f"Accuracy: {accuracy * 100}%")
