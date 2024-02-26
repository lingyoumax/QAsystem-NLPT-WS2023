import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import io
from tqdm import tqdm
try:
    from utils import jload,jdump
except:
    from utils.utils import jload,jdump
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, 
                        default="OpenAssistant/reward-model-deberta-v3-large-v2", help='Selection of score model')
    parser.add_argument('--quality_evaluation_file', type=str,
                         default="convert_data/merged_dataset.json", help='Select the file to be choosen high quanlity data')
    parser.add_argument('--save_file_path', type=str, default="argument_data/score_data.json", help='The file path to be save')
    args, unknown_args = parser.parse_known_args()

    return args

def exact_score(args=None,**kwargs):
    """
    args: have model_name quality_evaluation_file save_file_path
    if want to use function call, args should be None, and use kwargs to get parameters
    """
    if "args" in kwargs.keys():
        args=kwargs["args"]
        model_name=args.model_name
        quality_evaluation_file=args.quality_evaluation_file
        save_file_path=args.save_file_path
    else:
        try:
            model_name=kwargs["model_name"]
            quality_evaluation_file=kwargs["quality_evaluation_file"]
            save_file_path=kwargs["save_file_path"]
        except:
            raise ValueError("dont have model_name,quality_evaluation_file,save_file_path parameters")
        
    data=jload(quality_evaluation_file)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    rank_model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda()
    result_list=[]
    for element in tqdm(data):
        try:
            instruction=element["instruction"]
            input=element["input"]
            output=element["output"]
            question = instruction+"\n"+ input
            answer=output
            inputs = tokenizer(question, answer, return_tensors='pt').to("cuda")
            score = rank_model(**inputs).logits[0].detach()
            final_result = {'instruction':instruction,'input':input,'output':output,'reward_score':float(score)}
            result_list.append(final_result)
        except:
            continue
    print('number of data', len(result_list))
    
    jdump(result_list,save_file_path)

    print("successfull exact score!")

if __name__=="__main__":
    args=get_args()
    exact_score(args)
    