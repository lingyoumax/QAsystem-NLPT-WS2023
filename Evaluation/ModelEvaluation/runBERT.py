import json
from tqdm import tqdm
from getData import getData 
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from config import GLOBAL_CONFIG

# 加载预训练的模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# 输入的问题和参考内容

# 对输入进行编码
def generateAnswer(context,question):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")

# 获取输入的令牌类型ID，用于区分问题和内容
    token_type_ids = inputs["token_type_ids"]

# 获取输入的注意力掩码，指示模型哪些令牌是有意义的
    input_ids = inputs["input_ids"]

# 模型预测答案的起始和结束位置
    start_scores, end_scores = model(input_ids, token_type_ids=token_type_ids, return_dict=False)

# 确定答案的起始和结束位置
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

# 将令牌ID转换回文本
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    answer = ' '.join(all_tokens[start_index : end_index + 1])

# 输出答案
    return answer

data=getData()
num_ref=GLOBAL_CONFIG['num_ref']
num_rep=GLOBAL_CONFIG['num_rep']
qca=[]
for ref_index in tqdm(range(num_ref)):
    for i in range(num_rep):
        try:
            question = data[ref_index]['input']
            context = data[ref_index]['instruction']

            answer=generateAnswer(context,question)

            qca.append({"instruction":context,"input":question,"output":answer})
        except Exception as e:
            print(ref_index)
            print(e)

filename = 'Data/data_BERT.json'

# 打开文件用于写入，'w' 表示写入模式
with open(filename, 'w') as f:
    # 使用 json.dump 方法将数据写入文件
    # indent 参数是可选的，用于美化输出，使 JSON 文件易于阅读
    json.dump(qca, f, indent=4)