from transformers import AutoModelForCausalLM,BitsAndBytesConfig,AutoTokenizer
import torch.nn as nn
import torch
import os
from peft import prepare_model_for_kbit_training,PeftModel,LoraConfig,TaskType,get_peft_model
from utils.utils import find_all_linear_names

                
def find_v_head_weights_bin(directory,name):
    for root, dirs, files in os.walk(directory):
        if name in files:
            return os.path.join(root, name)
    return None

class QwenRM(nn.Module):
    def __init__(
            self,model_name,
            adapter=None,
    ):
        super().__init__()

        tokenizer= AutoTokenizer.from_pretrained(
            model_name,
            pad_token='<|extra_0|>',
            padding="left",
            eos_token='<|endoftext|>',
            trust_remote_code=True
            )

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
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)

        if adapter is not None:
            model=PeftModel.from_pretrained(model,adapter,is_trainable=True)
        else:
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

        self.tokenizer=tokenizer
        self.config=model.config
        self.config.n_embd=self.config.hidden_size if hasattr(self.config,"hidden_size") else self.config.n_embd
        self.transformer=model.transformer
        self.v_head=nn.Linear(self.config.n_embd,1,bias=False)
        self.tokenizer.pad_token=self.tokenizer.eos_token
        self.PAD_ID=151643
    
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        labels=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        loss = None
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]
        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_end_scores = []
        rejected_end_scores = []
        
        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        chosen = input_ids[:bs]
        rejected = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]
        
        loss = 0

        for i in range(bs):
            if self.tokenizer.padding_side == "left":
                start_ind = (chosen[i] != self.PAD_ID).nonzero().min().item() if (chosen[i] != self.PAD_ID).any() else 0
                divergence_ind = ((chosen[i] != rejected[i]) & (chosen[i] != self.PAD_ID)).nonzero().min().item() if ((chosen[i] != rejected[i]) & (chosen[i] != self.PAD_ID)).any() else start_ind
                end_ind = chosen.shape[1] 
            else:
                end_ind = (chosen[i] == self.PAD_ID).nonzero().min().item() if (chosen[i] == self.PAD_ID).any() else chosen.shape[1]
                divergence_ind = ((chosen[i] != rejected[i]) & (chosen[i] != self.PAD_ID)).nonzero().min().item() if ((chosen[i] != rejected[i]) & (chosen[i] != self.PAD_ID)).any() else 0
                start_ind = 0 

            if torch.all(torch.eq(chosen[i], rejected[i])).item():
                # 序列完全相同时的处理逻辑
                chosen_end_scores.append(chosen_rewards[i, end_ind - 1] if self.tokenizer.padding_side == "right" else chosen_rewards[i, start_ind])
                rejected_end_scores.append(rejected_rewards[i, end_ind - 1] if self.tokenizer.padding_side == "right" else rejected_rewards[i, start_ind])
            c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
            r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]
            chosen_end_scores.append(c_truncated_reward[-1])
            rejected_end_scores.append(r_truncated_reward[-1])
            
            # Compute loss based on truncated rewards (ignore padding)
            loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        loss = loss / bs
        
        chosen_end_scores = torch.stack(chosen_end_scores)
        rejected_end_scores = torch.stack(rejected_end_scores)
        
        
        return {
            "loss": loss,
            "chosen_end_scores": chosen_end_scores,
            "rejected_end_scores": rejected_end_scores,
        }

    def load_v_head_model(self, lora_path):
        """
        Load weights and bias into the v_head linear layer from separate files.

        Parameters:
        - weights_path: str, the path to the file containing the saved weights.
        - bias_path: str, the path to the file containing the saved bias.
        """
        v_head_weights_path=find_v_head_weights_bin(lora_path,"v_head_weights.bin")
        v_head_bias_path=find_v_head_weights_bin(lora_path,"v_head_bias.bin")
        v_head_weights = torch.load(v_head_weights_path)
        if v_head_bias_path is not None:
            v_head_bias = torch.load(v_head_bias_path)
        if hasattr(self, 'v_head'):
            self.v_head.weight.data.copy_(v_head_weights)
            if v_head_bias_path is not None:
                self.v_head.bias.data.copy_(v_head_bias)
        else:
            print("not find v_head, initial from the v_head_weights.bin")
            if v_head_bias_path is not None:
                self.v_head = nn.Linear(v_head_weights.size(1), v_head_weights.size(0), bias=True)
            else:
                self.v_head = nn.Linear(v_head_weights.size(1), v_head_weights.size(0), bias=False)
            self.v_head.weight.data.copy_(v_head_weights)
            if v_head_bias_path is not None:
                self.v_head.bias.data.copy_(v_head_bias)
