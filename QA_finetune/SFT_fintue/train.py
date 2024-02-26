from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from datasets import Dataset
import torch
from transformers import  AutoModelForCausalLM, DataCollatorForSeq2Seq,BitsAndBytesConfig,Trainer
from peft import LoraConfig, TaskType, get_peft_model,PeftModel,prepare_model_for_kbit_training 
import pandas as pd
import os
import global_config
from utils.utils import process_func,find_all_linear_names

class ModifiedTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


if "__main__" == __name__:
    # 处理数据集
    # 将JSON文件转换为CSV文件
    writer = SummaryWriter()
    df = pd.read_json('data/finetune_dataset_cleaned.json')
    ds = Dataset.from_pandas(df)
    # 加载tokenizer
    
    # 将数据集变化为token形式
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
    bnb_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    model = AutoModelForCausalLM.from_pretrained(global_config.model_name, trust_remote_code=True, quantization_config=bnb_config, device_map="auto")
    model.supports_gradient_checkpointing = True
    model.gradient_checkpointing_enable()
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
    trainer = ModifiedTrainer(
        model=model,
        args=global_config.args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=global_config.tokenizer, padding=True),
        callbacks=[TensorBoardCallback(writer)],
        )
    trainer.train() # 开始训练

    writer.close()
    # save model
    model.save_pretrained(global_config.args.output_dir)