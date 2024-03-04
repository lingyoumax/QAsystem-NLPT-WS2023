from transformers import AutoModel, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Qwen/Qwen1.5-0.5B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

def getStability(sentence1, sentence2):
    if len(sentence1)==0 or len(sentence2)==0:
        return 1
    inputs1 = tokenizer(sentence1, return_tensors="pt", padding=True, truncation=True).to(device)
    inputs2 = tokenizer(sentence2, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs1 = model(**inputs1).last_hidden_state
        outputs2 = model(**inputs2).last_hidden_state

    features1 = outputs1[:, 0, :]
    features2 = outputs2[:, 0, :]

    cosine_similarity = torch.nn.functional.cosine_similarity(features1, features2)
    return cosine_similarity.item()

