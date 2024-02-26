'''
    方法1：语法检查
    方法2: 自然度和流畅性 perplexity 文本困惑度 这里对文本困惑度 可以有其他更深层次的应用 很多可以玩的
'''
import language_tool_python
import string


def grammar_check(text):
    # 删除标点符号
    text_nopunct = text.translate(str.maketrans('', '', string.punctuation))

    # 初始化LanguageTool
    tool = language_tool_python.LanguageTool('en-US')

    # 对无标点文本进行语法检查
    matches = tool.check(text_nopunct)

    # 如果找到匹配项，打印错误和建议
    if matches:
        for match in matches:
            print(f"Error: {match.matchedText}")
            print(f"Suggestions: {', '.join(match.replacements)}")
            print(f"Context: {match.context}\n")
    else:
        print("No grammar mistakes found.")


# 示例文本
text = "Some challenges and limitations include the reliability of medical data, limitations around predictive modeling, and data sharing between healthcare organizations."

# 调用函数进行拼写检查
grammar_check(text)

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


def check_fluency(text):
    # 加载预训练的GPT-2模型及其分词器
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # 编码文本，添加必要的特殊令牌
    inputs = tokenizer.encode(text, return_tensors='pt')

    # 使用GPT-2模型预测文本的下一个词
    outputs = model(inputs, labels=inputs)
    loss, logits = outputs[:2]

    # 计算并返回困惑度（Perplexity）
    perplexity = torch.exp(loss).item()
    return perplexity


# 调用函数进行自然度和流畅性检查
perplexity = check_fluency(text)
print(f"Perplexity: {perplexity}")
