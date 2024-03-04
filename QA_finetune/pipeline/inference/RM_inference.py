from model.Qwen_rm import QwenRM
from tqdm import tqdm
import torch
import gc

def inference(instructions,model_name,adapter,batch_size=2,device="cuda"):
    model=QwenRM(model_name,adapter).eval()
    if adapter is not None:
        model.load_v_head_model(adapter)
    tokenizer= model.tokenizer
    
    model.to(device)
    
    result_instructions=[]
    for index in tqdm(range(0,len(instructions),batch_size)):
        start=index
        end=min(start+batch_size,len(instructions))

        batch={}
        chosen_input_ids=[]
        chosen_attn_masks=[]
        rejected_input_ids=[]
        rejected_attn_masks=[]
        for j in range(start,end):
            if "instruction" in instructions[j].keys():
                prompt="\n".join(["<|im_start|>system", "Now you are a question answering assistant.<|im_end|>" + "\n<|im_start|>user\n" +"Based on content:\n"+ instructions[j]["instruction"]+"Answer Question:\n" + instructions[j]["input"] + "<|im_end|>\n"])
                chosen=prompt+"\n"+instructions[j]["generated"]
                reject=prompt+"\n"+instructions[j]["generated"]
                
            else:
                prompt=instructions[j]["prompt"]
                chosen=instructions[j]["answer"]
                reject=instructions[j]["answer"]
            chosen_encodings_dict = tokenizer(
                    "<|startoftext|>" + chosen + "<|endoftext|>",
                    truncation=True,
                    max_length=649,
                    padding="max_length",
                    return_tensors="pt",
                )
            rejected_encodings_dict = tokenizer(
                    "<|startoftext|>" + reject + "<|endoftext|>",
                    truncation=True,
                    max_length=649,
                    padding="max_length",
                    return_tensors="pt",
                )

            chosen_input_ids.append(chosen_encodings_dict["input_ids"])
            chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
            rejected_input_ids.append(rejected_encodings_dict["input_ids"])
            rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])

        batch = {}
        batch["input_ids"] = torch.cat(chosen_input_ids+rejected_input_ids).to(device)
        batch["attention_mask"] = torch.cat(chosen_attn_masks+rejected_attn_masks).to(device)
        batch["labels"] = torch.tensor([0] * len(chosen_input_ids) + [1] * len(rejected_input_ids)).to(device)
        with torch.no_grad():
            output=model(**batch)
        for j in range(start,end):
            try:
                if hasattr(instructions[j],"output"):
                    result_instruction = {'index':instructions[j]["index"] if "index" in instructions[j].keys() else j,'instruction':instructions[j]["instruction"], 'input':instructions[j]["input"],
                                        'output':instructions[j]["output"], 'generated':instructions[j]["generated"],
                                        'score':output['chosen_end_scores'][j-start].item()}
                else:
                    result_instruction = {'index':instructions[j]["index"] if "index" in instructions[j].keys() else j,'instruction':instructions[j]["instruction"], 
                                        'input':instructions[j]["input"],'generated':instructions[j]["generated"],
                                        'score':output['chosen_end_scores'][j-start].item()}
            except:
                result_instruction = {'index':instructions[j]["index"] if "index" in instructions[j].keys() else j,"prompt":instructions[j]["prompt"],
                                    "answer":instructions[j]["answer"],
                                    'score':output['chosen_end_scores'][j-start].item()}
            result_instructions.append(result_instruction)
    print("success inference!")
    del model
    torch.cuda.empty_cache()  # 告诉CUDA释放尽可能多的显存
    gc.collect()
    return result_instructions