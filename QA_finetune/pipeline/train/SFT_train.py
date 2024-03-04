from transformers import Trainer,AutoTokenizer,BitsAndBytesConfig,DataCollatorForSeq2Seq,AutoModelForCausalLM
import os
import torch
from datasets import Dataset
from peft import (prepare_model_for_kbit_training,LoraConfig
                  ,TaskType,get_peft_model,PeftModel)
from utils.utils import  find_all_linear_names

def create_process_func(tokenizer):
    def process_func(sample):
        MAX_LENGTH = 1024 
        input_ids, attention_mask, labels = [], [], []
        instruction = tokenizer("\n".join(["<|im_start|>system", "Now you are a question answering assistant.<|im_end|>" + "\n<|im_start|>user\n" +"Based on content:\n"+ sample["instruction"]+"Answer Question:\n" + sample["input"] + "<|im_end|>\n"]).strip(), add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
        response = tokenizer("<|im_start|>assistant:\n" + sample["output"] + "<|im_end|>\n", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  # Qwen的特殊构造就是这样的
        if len(input_ids) > MAX_LENGTH:  # 做一个截断
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    return process_func

class ModifiedTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


def train(model_name,adapter=None,dataset_path=None,
          train_args=None,save_dir=None):
    bnb_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,use_flash_attn=False, quantization_config=bnb_config, device_map="auto")
    model.enable_input_require_grads()
    #inference should be true
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    
    tokenizer= AutoTokenizer.from_pretrained(model_name,trust_remote_code=True) 
    tokenizer.pad_token_id = tokenizer.eod_id
    print("initial train model")
    
    process_func = create_process_func(tokenizer)
    
    dataset=Dataset.from_json(dataset_path)
    dataset=dataset.map(process_func, remove_columns=dataset.column_names)
    
    if adapter is None:
        lora_modules = find_all_linear_names(model)
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            target_modules=lora_modules,  
            inference_mode=False,
            r=8,
            lora_alpha=32, 
            lora_dropout=0.1
        )
        model = get_peft_model(model, config)
    else:
        model=PeftModel.from_pretrained(model,adapter,is_trainable=True)
    model.print_trainable_parameters()

    trainer = ModifiedTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        )
    trainer.train()

    trainer.save_model(save_dir)
    
