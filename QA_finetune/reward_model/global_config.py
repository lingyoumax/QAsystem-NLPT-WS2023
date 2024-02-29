model_name="/root/Qwen_7B"
lora_model="/root/QA_Qwen/Qwen_sft_out"
from transformers import AutoTokenizer,TrainingArguments

tokenizer = AutoTokenizer.from_pretrained(
model_name,
pad_token='<|extra_0|>',
padding="left",
eos_token='<|endoftext|>',
trust_remote_code=True
)

run_name="Reward_fintune_v1"
output_dir="Qwen-rm-v1"
train_args = TrainingArguments(
    report_to="wandb",
    run_name=run_name,
    output_dir=output_dir,
    per_device_train_batch_size=6,
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