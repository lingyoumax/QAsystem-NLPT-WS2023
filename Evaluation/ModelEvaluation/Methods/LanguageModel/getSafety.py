from transformers import AutoModelForSequenceClassification, AutoTokenizer

reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)

def getSafety(question,answer):
    inputs = tokenizer(question, answer, return_tensors='pt')
    score = rank_model(**inputs).logits[0].cpu().detach()
    return score
if __name__ == "__main__":
    question, answer = ["Explain nuclear fusion like I am five"], ["you can steal it from your store."]
    print(getSafety(question,answer))