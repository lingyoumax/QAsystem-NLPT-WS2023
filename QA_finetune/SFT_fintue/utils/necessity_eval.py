import argparse
try:
    from utils.utils import jload,jdump
except:
    from utils import jload,jdump
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

def necessity_eval(**kwargs):
    """
    kwargs should have args or [no_k_greedy_path,model_name,save_file_path]
    """
    if "args" in kwargs.keys():
        args=kwargs["args"]
        no_k_greedy_path=args.no_k_greedy_path
        model_name=args.model_name
        save_file_path=args.save_file_path
    else:
        try:
            no_k_greedy_path=kwargs["no_k_greedy_path"]
            model_name=kwargs["model_name"]
            save_file_path=kwargs["save_file_path"]
        except:
            raise ValueError("dont have [no_k_greedy_path,model_name,save_file_path] parameters")
    
    no_k_greedy_list=jload(no_k_greedy_path)
    print('number of input file', len(no_k_greedy_list))
    rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(model_name).cuda(), AutoTokenizer.from_pretrained(model_name)

    result_list = []
    for element in tqdm(no_k_greedy_list):
        instruction = element['instruction']
        input = element['input']
        output = element['output']
        generated = element['generated']
    
        question = instruction+'\n'+input
        
        answer = generated
        
        try:
            inputs = tokenizer(question, answer, return_tensors='pt').to("cuda")
            score = rank_model(**inputs).logits[0].detach()
            final_result = {'instruction':instruction,'input':input,'output':output,'generated':generated,'reward_score':float(score)}
            result_list.append(final_result)
        except:
            print(instruction)
            print(generated)
            continue

    print('number of data', len(result_list))

    jdump(result_list,save_file_path)
    print("successful score generated data!")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_k_greedy_path', type=str, default="argument_data/no_k_greedy_generate.json", help='choose no greedy dataset')
    parser.add_argument('--model_name', type=str,
                         default="OpenAssistant/reward-model-deberta-v3-large-v2", help='reward model name')
    parser.add_argument('--save_file_path', type=str, default="argument_data/no_k_greedy_generate_score.json", help='save no greedy score dataset')
    args, unknown_args = parser.parse_known_args()

    return args

if __name__=="__main__":
    no_k_greedy_path="argument_data/no_k_greedy_generate.json"
    model_name="OpenAssistant/reward-model-deberta-v3-large-v2"
    save_file_path="argument_data/no_k_greedy_generate_score.json"
    necessity_eval(no_k_greedy_path=no_k_greedy_path,model_name=model_name,save_file_path=save_file_path)