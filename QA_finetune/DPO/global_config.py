model_name="/root/Qwen_7B"
lora_model="Qwen_sft_out"
from transformers import AutoTokenizer,TrainingArguments

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

run_name="DPO_fintune_v1"
output_dir="Qwen-DPO-v1"
train_args = TrainingArguments(
    report_to="wandb",
    run_name=run_name,
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    logging_strategy="steps",
    logging_steps=10,
    remove_unused_columns=False,
    num_train_epochs=1,
    save_strategy="steps",
    save_steps=50,
    learning_rate=1e-4,
    save_total_limit=1,
)

beta=0.2