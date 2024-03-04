import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.tokenize import word_tokenize
import nltk
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score

nltk.download('wordnet')
nltk.download('punkt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, doc, label = self.data[idx]
        query_enc = self.tokenizer(query, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        doc_enc = self.tokenizer(doc, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        return query_enc, doc_enc, torch.tensor(label, dtype=torch.float32)


class TextSummarizationAgent:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5)  # 可以调整学习率

    def summarize_text(self, text, max_length=512):
        preprocess_text = text.strip().replace("\n", "")
        t5_prepared_Text = "summarize: " + preprocess_text

        tokenized_text = self.tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

        summary_ids = self.model.generate(tokenized_text,
                                          num_beams=4,
                                          no_repeat_ngram_size=2,
                                          min_length=30,
                                          max_length=max_length,
                                          early_stopping=True)

        output = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return output



    def update(self, query, doc, reward):

        inputs = self.tokenizer.encode("summarize: " + doc, return_tensors='pt').to(self.device)
        outputs = self.model(input_ids=inputs, labels=inputs)
        logits = outputs.logits

        loss = F.cross_entropy(logits.view(-1, self.model.config.vocab_size), inputs.view(-1))
        policy_gradient = -loss * reward

        self.optimizer.zero_grad()
        policy_gradient.backward()
        self.optimizer.step()


def calculate_reward(candidate, reference):
    def calculate_rouge_scores(candidates, references):
        rouge = Rouge()
        scores = rouge.get_scores(candidates, references, avg=True)
        return scores

    def calculate_meteor_score(candidate, reference):
        candidate_tokens = word_tokenize(candidate)
        reference_tokens = word_tokenize(reference)

        return meteor_score([reference_tokens], candidate_tokens)

    meteor = calculate_meteor_score(candidate, reference)

    rouge_scores = calculate_rouge_scores([candidate], [reference])
    rouge_1 = rouge_scores['rouge-1']['f']
    rouge_2 = rouge_scores['rouge-2']['f']
    rouge_l = rouge_scores['rouge-l']['f']

    rouge_average = (rouge_1 + rouge_2 + rouge_l) / 3

    reward = 0.5 * meteor + 0.5 * rouge_average

    return reward


def train(model, tokenizer, dataset, device, num_epochs=2):  
    agent = TextSummarizationAgent(model, tokenizer, device)
    rewards = []  

    for epoch in range(num_epochs):
        total_reward = 0
        num_samples = 0

        data_loader = DataLoader(dataset, batch_size=64)
        for query_enc, doc_enc, _ in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            for doc_id_tensor in doc_enc['input_ids']:
                doc_id_list = doc_id_tensor.tolist()
                if isinstance(doc_id_list[0], list):
                    doc_id_list = doc_id_list[0]
                doc = tokenizer.decode(doc_id_list, skip_special_tokens=True)
                summarized_text = agent.summarize_text(doc)
                reward = calculate_reward(summarized_text, doc)
                agent.update(query_enc['input_ids'].to(device), doc, reward)

                total_reward += reward
                num_samples += 1
                if num_samples % 100 == 0:
                    print(f"Iteration {num_samples}: Current Reward = {reward}")

        average_reward = total_reward / num_samples
        rewards.append(average_reward)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Reward: {average_reward}")

    rewards_df = pd.DataFrame({'Epoch': range(1, num_epochs + 1), 'Average Reward': rewards})

    plt.figure(figsize=(10, 6))
    plt.plot(rewards_df['Epoch'], rewards_df['Average Reward'], marker='o')
    plt.title('Average Reward per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.show()

    model_save_path = 'trained_t5_model.pth'
    torch.save(agent.model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return rewards

model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model = model.to(device)

file_path_qap = 'qap.csv'
df = pd.read_csv('./PubmedDataSet.csv')
df['combined'] = df['Abstract'].fillna("") + " " + df['PubDate'].fillna("") + " " + df['Authors'].fillna("")
dataset = QAPDataset(file_path_qap, tokenizer, df)


train(model, tokenizer, dataset, device)
