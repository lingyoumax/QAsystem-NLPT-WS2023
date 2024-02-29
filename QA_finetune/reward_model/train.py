from datasets import Dataset
from dataloader import load_dataset,DataCollatorReward,PairwiseDataset
from transformers import Trainer
from utils import compute_metrics
import os
from Qwen_rm import QwenRM
import torch
import global_config

if not os.path.exists("rm_checkpoint"):
    os.mkdir("rm_checkpoint")

def load_model():
    if global_config.lora_model!="":
        model=QwenRM(global_config.lora_model)
    else:
        model=QwenRM()

    return model

def load_convert_dataset(path):
    data=load_dataset(path)
    dataset=PairwiseDataset(data,global_config.tokenizer,max_length=649)
    return dataset

class ModifiedTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))
        
        if hasattr(self.model, 'v_head'):
            v_head_weights = self.model.v_head.weight.data.cpu()
            torch.save(v_head_weights, os.path.join(output_dir, "v_head_weights.bin"))
            
            v_head_bias = self.model.v_head.bias.data.cpu()
            torch.save(v_head_bias, os.path.join(output_dir, "v_head_bias.bin"))
        
        
if __name__=="__main__":
    model=load_model()
    tokenizer=global_config.tokenizer

    train_dataset=load_convert_dataset("train.json")
    test_dataset=load_convert_dataset("test.json")

    trainer = ModifiedTrainer(
            model=model,
            args=global_config.train_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorReward(),
            )
    trainer.train()
    model.save_pretrained(global_config.output_dir + "_peft_last_checkpoint")
