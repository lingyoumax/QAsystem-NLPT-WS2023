import os

import json
import sys
import numpy as np
from transformers import BertTokenizer, BertModel,AutoModel
import torch
from kcenter_greedy import *
import argparse

@torch.no_grad()
def bert_embedding(texts,batch=100):
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
    model = AutoModel.from_pretrained('google-bert/bert-base-uncased').cuda()
    # 将文本转化为BERT模型可识别的token序列
    encoded_texts = tokenizer(texts,return_tensors="pt",truncation=True,padding=True,max_length=512)
    encoded_texts =  encoded_texts.to("cuda")
    cls_hid_li = []
    # 使用BERT模型对每个文本序列进行编码,提取其语义向量
    i= 0
    while i < len(texts):
        last_hids = model(input_ids=encoded_texts["input_ids"][i:i+batch],
                          attention_mask=encoded_texts["attention_mask"][i:i+batch])['last_hidden_state']
        cls_hids = last_hids[:,0,:].squeeze()
        cls_hid_li.append(cls_hids)
        i+= batch
        print(i)
    # 将所有文本的embedding连成特征矩阵
    cls_hids_tensor = torch.concat(cls_hid_li, dim=0)
    np.save("bert_embedding.npy",cls_hids_tensor.cpu())
    return np.array(cls_hids_tensor.cpu())

# 数据采样
def sample_func(text_list,K):
    result = []
    if os.path.exists("bert_embedding.npy"):
        text_embedding = np.load("bert_embedding.npy")
    else:
        text_embedding = bert_embedding(text_list)
        np.save("bert_embedding.npy",text_embedding)
    
    result = []

    k_center = kCenterGreedy(text_embedding)
    
    already_selected = None
    #for _ in range(K):
    result = k_center.select_batch_(text_embedding,already_selected,K)
        #result = result + new_data
        #already_selected += new_data
    return result


def k_greedy(**kwargs):
    """
    should have args or [high_quality_data_path,k_greedy_save_path,top_k]
    """
    if "args" in kwargs.keys():
        args=kwargs["args"]
        high_quality_data_path=args.high_quality_data_path
        k_greedy_save_path=args.k_greedy_save_path
        top_k=args.top_k
    else:
        try:
            high_quality_data_path=kwargs["high_quality_data_path"]
            k_greedy_save_path=kwargs["k_greedy_save_path"]
            top_k=kwargs["top_k"]
        except:
            raise ValueError("dont have [high_quality_data_path,k_greedy_save_path,top_k] param")

    data = json.load(fp=open(high_quality_data_path, "r"))
    instruction_list = []
    for d in data:
        instruction_list.append(d["instruction"])
    res = sample_func(text_list = instruction_list, K = top_k)
    print('data length')
    print(len(data))
    
    print('sampling data:')
    print(len(res))
    print(res)
    data_li = []
    for index in res:
        data_li.append(data[index])
    json.dump(obj=data_li,fp=open(k_greedy_save_path,"w"),indent=2,ensure_ascii=False)
    print("successfull k greedy!")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, 
                        default="2000", help='top_k different data')
    parser.add_argument('--high_quality_data_path', type=str,
                         default="argument_data/high_quality.json", help='Select the file to be judged')
    parser.add_argument('--k_greedy_save_path', type=str, default="argument_data/k_greedy.json", help='Select the file to be judged')
    args, unknown_args = parser.parse_known_args()

    return args

if __name__ == "__main__":
    args=get_args()
    k_greedy(args=args)