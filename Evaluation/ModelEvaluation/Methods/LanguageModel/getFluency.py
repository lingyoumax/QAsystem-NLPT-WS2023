from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def getFluency(text):
    # 加载预训练的GPT-2模型及其分词器
    if len(text)==0:
        return 0
    
    # 编码文本，添加必要的特殊令牌
    inputs = tokenizer.encode(text, return_tensors='pt')
    if inputs.numel()==0:
        return 0

    # 使用GPT-2模型预测文本的下一个词
    outputs = model(inputs, labels=inputs)
    loss, logits = outputs[:2]

    # 计算并返回困惑度（Perplexity）
    perplexity = torch.exp(loss).item()
    return perplexity