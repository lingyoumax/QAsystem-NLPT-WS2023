from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
from getData import getData
from config import GLOBAL_CONFIG

prompt_template = """Answer the following QUESTION based on the CONTEXT
    given. If you do not know the answer and the CONTEXT doesn't
    contain the answer truthfully say "I don't know".

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """

model_path = '01-ai/Yi-6b-Chat'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
max_length=600

# Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()

def generateAnswer(context,question):

    text_input = prompt_template.replace("{context}", context).replace("{question}", question)
    messages = [
        {"role": "user", "content": text_input}
    ]

    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt',max_length=max_length)
    output_ids = model.generate(input_ids.to('cuda'),max_length=max_length)
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response

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

filename = 'Data/data_Yi.json'

# 打开文件用于写入，'w' 表示写入模式
with open(filename, 'w') as f:
    # 使用 json.dump 方法将数据写入文件
    # indent 参数是可选的，用于美化输出，使 JSON 文件易于阅读
    json.dump(qca, f, indent=4)