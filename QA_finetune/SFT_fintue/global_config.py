model_name="qwen/Qwen-7B-Chat"
lora_model=""

from transformers import AutoTokenizer
from transformers import TrainingArguments
tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eod_id

args = TrainingArguments(
    output_dir="./output/Qwen",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_strategy="steps",
    logging_steps=1,
    num_train_epochs=3,
    gradient_checkpointing=True,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    save_total_limit=2,
)