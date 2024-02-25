# 1. Base model and training environment
- base model: RWKV-5-wolrd-1.5b
- GPU: 4090,24GB
- Fine-tuning method：LoRA
# 2.Current progress
Currently, due to resource issues, fine-tuning training of 1.5b is attempted. After further considering the resource issue, we chose the LoRA fine-tuning method, which can be seen for details (https://huggingface.co/docs/transformers/main/zh/peft). Preliminary fine-tuning is currently underway.
# 3. Next goal
- Further fine-tuning
- Try using a larger model but fine-tuning it using QLoRA.
- TrainingReward model
- Use RLHF method to further fine-tune
- Evaluation model
# 4.How to run it
Currently, the pre-trained LoRA model needs to be downloaded locally (https://drive.google.com/drive/folders/1N3rduBWmft_LLJbZSkeB1PLl7lYTjzDG?usp=drive_link), modify the path of the lora model in chat.py, and run chat.py to use it.
(It should be noted that the peft version needs to be above 0.5.0 to support CPU inference.)
