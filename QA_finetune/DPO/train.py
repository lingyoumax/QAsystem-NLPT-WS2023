from dataloader import load_dataset
from datasets import Dataset
from transformers import Trainer,AutoModelForCausalLM,BitsAndBytesConfig
from peft import prepare_model_for_kbit_training,LoraConfig,TaskType,get_peft_model,PeftModel
from utils import find_all_linear_names
import os
import torch
import global_config
from trl import DPOTrainer

if not os.path.exists("rm_checkpoint"):
    os.mkdir("rm_checkpoint")

def load_model():
    bnb_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    model = AutoModelForCausalLM.from_pretrained(global_config.model_name, trust_remote_code=True, use_flash_attn=False,quantization_config=bnb_config, device_map="auto")
    model.enable_input_require_grads()
    #inference should be true
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    if global_config.lora_model=="":
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
        model=PeftModel.from_pretrained(model,global_config.lora_model,is_trainable=True)
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
        
if __name__=="__main__":
    model=load_model()
    tokenizer=global_config.tokenizer

    train_dataset=load_convert_dataset("train.json")
    test_dataset=load_convert_dataset("test.json")

    trainer = ModifiedTrainer(
            model=model,
            ref_model=None,
            args=global_config.train_args,
            beta=global_config.beta,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=global_config.tokenizer,
            max_prompt_length=512,
            max_length=1024,
            )
    trainer.train()
    trainer.save_model(global_config.output_dir+"/DPO_out")
    model.save_pretrained(global_config.output_dir + "_peft_last_checkpoint")
