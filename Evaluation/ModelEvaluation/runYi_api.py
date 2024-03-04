import replicate
import os
import json
from tqdm import tqdm
from getData import getData
from config import GLOBAL_CONFIG

os.environ["REPLICATE_API_TOKEN"] = GLOBAL_CONFIG['replicate_key']
def generateAnswer(context,question):
    prompt_template = """Answer the following QUESTION based on the CONTEXT
    given. If you do not know the answer and the CONTEXT doesn't
    contain the answer truthfully say "I don't know".

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """

    text_input = prompt_template.replace("{context}", context).replace("{question}", question)
    output = replicate.run(
        "01-ai/yi-34b-chat:914692bbe8a8e2b91a4e44203e70d170c9c5ccc1359b283c84b0ec8d47819a46",
        input={
            "top_k": 50,
            "top_p": 0.8,
            "prompt": text_input,
            "temperature": 0.3,
            "max_new_tokens": 1024,
            "prompt_template": "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            "repetition_penalty": 1.2
        }
    )

    # The 01-ai/yi-34b-chat model can stream output as it's running.
    # The predict method returns an iterator, and you can iterate over that output.
    answer=""
    for item in output:
        # https://replicate.com/01-ai/yi-34b-chat/api#output-schema
        answer=answer+item
    
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

filename = 'Data/data_Yi.json'

# 打开文件用于写入，'w' 表示写入模式
with open(filename, 'w') as f:
    # 使用 json.dump 方法将数据写入文件
    # indent 参数是可选的，用于美化输出，使 JSON 文件易于阅读
    json.dump(qca, f, indent=4)