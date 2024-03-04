from datasets import Dataset
from transformers import AutoModelForCausalLM,BitsAndBytesConfig,AutoTokenizer
from peft import prepare_model_for_kbit_training,LoraConfig,TaskType,get_peft_model,PeftModel
import os
import torch
from trl import DPOTrainer
from utils.utils import jload,find_all_linear_names
import gc

def load_dataset(path):
    dataset=jload(path)
    pairs={"prompt":[],"chosen":[],"rejected":[]}
    for sample in dataset:
        prompt=sample.get("prompt",None)
        chosen_summary=sample.get("chosen",None)
        rejected_summary=sample.get("rejected",None)
        if prompt is None or chosen_summary is None or rejected_summary is None:
            continue
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue
        pairs["chosen"].append(chosen_summary)
        pairs["rejected"].append(rejected_summary)
        pairs["prompt"].append(prompt)
    print("actual create {} num samples".format(len(pairs)))
    return pairs

def load_model(model_name,adapter=None,):
    bnb_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, use_flash_attn=False,quantization_config=bnb_config, device_map="auto")
    model.enable_input_require_grads()
    #inference should be true
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    if adapter is None:
        lora_modules = find_all_linear_names(model)
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            target_modules=lora_modules,  # 这个不同的模型需要设置不同的参数，需要看模型中的attention层
            inference_mode=False, # 训练模式
            r=8, # Lora 秩
            lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
            lora_dropout=0.1# Dropout 比例
        )
        model = get_peft_model(model, config)
    else:
        model=PeftModel.from_pretrained(model,adapter,is_trainable=True)
    model.print_trainable_parameters()
    
    return model

def load_convert_dataset(path):
    data=load_dataset(path)
    dataset=Dataset.from_dict(data)
    
    def return_prompt_and_responses(samples):
        return {
            "prompt": samples["prompt"],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
        }

    return dataset.map(return_prompt_and_responses,
                       batched=True)

class ModifiedTrainer(DPOTrainer):
    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))
        
def train(model_name,adapter=None,dataset_path=None,
          train_args=None,save_dir=None,beta=0.1):
    train_dataset=load_convert_dataset(dataset_path)
    
    model=load_model(model_name,adapter)

    tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            pad_token='<|extra_0|>',
            padding="left",
            eos_token='<|endoftext|>',
            trust_remote_code=True
            )
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    

    trainer = ModifiedTrainer(
            model=model,
            ref_model=None,
            args=train_args,
            beta=beta,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            max_prompt_length=512,
            max_length=1024,
            )
    trainer.train()
    trainer.save_model(save_dir)
    model.save_pretrained(save_dir)
    del model
    torch.cuda.empty_cache()  
    gc.collect()
