### About pipeline
This pipeline is further fine-tuned based on SFT_v0 and rm_v0.

### About the role of each folder
- data:Store intermediate temporary files and the final train file
- inference:Function files used for inference
- lora_out:Store each fine-tuning adapter.
- model:Store reward model schema file
- seed_question:Store the seed question and the generated question dataset
- train:Includes DPO training files and SFT training files

### file information
- About the actual content of lora_out: [rm_out](https://drive.google.com/drive/folders/1jpcBMJ_tisyFxn2j8qMjY3M4Nm98gUBu?usp=drive_link),[SFT_out](https://drive.google.com/drive/folders/1L6feQcrPOoORwVS9j0UsujLYZbetOr6G?usp=drive_link)
- About the actual content of seed question:need add [filterchunk](https://drive.google.com/file/d/1QXCMTbAqbG4kYhL0nsjPGPyvj0rzGfWe/view?usp=drive_link)
- About the actual content of data: This only includes the final train data, which is applicable to DPO and reject sampling respectively. If you need intermediate data, please contact us.

### how to use
- Run the files using DPO.py and reject_sampling.py respectively.
