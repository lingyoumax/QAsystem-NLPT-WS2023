from utils.utils import jload,jdump
from utils.generate_question import generate_question

def DPO(finetune_config):
    """
    finetune_steps:[1,2,3,4], can choose any one of them to indicate which steps to perform. such as [1] means only generate seed questions
    """
    if "finetune_steps" not in finetune_config.keys():
        raise ValueError("should Need to specify the order of fine-tuning")
    finetune_steps=finetune_config["finetune_steps"]

    for step in finetune_steps:
        if not isinstance(step,int):
            raise ValueError("step should be interger.")
        if step<0 or step>4:
            raise ValueError("step should be in range(0,4)")
    
    if 1 in finetune_steps:
        print("step 1:Execute seed question generation.")

        if (batch_dir := finetune_config.get("seed_question_dir")) is None:
            raise ValueError("not find seed_question_dir")
        
        if (seed_tasks_path := finetune_config.get("seed_question_output_dir")) is None:
            raise ValueError("not find seed_question_output_dir")
        
        if (platform := finetune_config.get("platform") ) is None or (engine:= finetune_config.get("engine"))is None:
            platform="GPT3"
            engine="gpt-3.5-turbo-instruct"
            print("not find platform or engine API setting in confing, use GPT3 and gpt-3.5-turbo-instruct.")
        
        if (num_prompt_instructions := finetune_config.get("num_prompt_instructions")) is None:
            num_prompt_instructions=8
            print("not find num_prompt_instructions, use default 8")
        
        if (num_instructions_to_generate := finetune_config.get("num_instructions_to_generate")) is None:
            num_instructions_to_generate=2000
            print("not find num_instructions_to_generate, use default 2000")

        if (request_batch_size := finetune_config.get("request_batch_size")) is None:
            request_batch_size=5
            print("not find request_batch_size, use default 5")
        generate_question(batch_dir,seed_tasks_path,platform,engine,num_prompt_instructions,num_instructions_to_generate,request_batch_size)

        print("saved generate question file in {}/machine_generated_instructions.json".format(seed_tasks_path))

    if 2 in finetune_steps:
        from inference.SFT_inference import inference as SFT_inference
        print("step 2:Generate responses using two pretrained model to combine pairs.")

        if (instruction_dir := finetune_config.get("instruction_dir")) is None:
            raise ValueError("not find instruction_dir")
        
        instruction=jload(instruction_dir)

        if (SFT_model := finetune_config.get("generateion_question_base_model")) is None:
            raise ValueError("not find generateion_question_base_model")
        
        if (SFT_lora_out_inference := finetune_config.get("generateion_question_lora_model")) is None:
            print("not find generateion_question_lora_model, use base model")

        if (SFT_model_2 := finetune_config.get("generateion_question_base_model_2")) is None:
            raise ValueError("not find generateion_question_base_model 2")
        
        if (SFT_lora_out_inference_2 := finetune_config.get("generateion_question_lora_model_2")) is None:
            print("not find generateion_question_lora_model_2, use base model 2")
        
        if (response_path := finetune_config.get("response_path")) is None:
            response_path="data/response_result.json"
            print("not find response_path, use data/response_result.json")
        
        if (response_num := finetune_config.get("response_num")) is None:
            response_num=2
            print("not find response_num, use 2")

        generate_result=SFT_inference(instruction,SFT_model,SFT_lora_out_inference,response_num=response_num)
        generate_result_2=SFT_inference(instruction,SFT_model_2,SFT_lora_out_inference_2,response_num=response_num)
        res=generate_result+generate_result_2
        jdump(res,response_path)

    if 3 in finetune_steps:
        from inference.RM_inference import inference as RM_inference
        print("step 3: score and combine chosen rejected pairs")

        if (RM_model := finetune_config.get("score_base_model")) is None:
            raise ValueError("not find score_model")
        
        if (RM_lora_out_inference := finetune_config.get("score_lora_model")) is None:
            raise ValueError("not find score_lora_model")
        
        if (response_dir := finetune_config.get("response_dir")) is None:
            if 2 in finetune_steps:
                instruction=generate_result
                print("use the result file in step 2")
            else:
                raise ValueError("not find response_dir")
        else:
            instruction=jload(response_dir)

        if (score_path := finetune_config.get("score_path")) is None:
            score_path="data/score_result.json"
            print("not find score_path, use data/score_result.json")

        score_result=RM_inference(instruction,RM_model,RM_lora_out_inference)
        jdump(score_result,score_path)

        from utils.utils import combine_pair_response
        score_result=jload(score_path)
        pairs_json=combine_pair_response(score_result)
        if (pairs_path := finetune_config.get("pairs_path")) is None:
            pairs_path="data/pairs_path.json"
            print("not find pairs_path, use data/pairs_path.json")
        jdump(pairs_json,pairs_path)
    
    if 4 in finetune_steps:
        print("step 4: DPO finetune step")

        from train.DPO_train import train
        from transformers import TrainingArguments

        if (SFT_model := finetune_config.get("finetune_model")) is None:
            raise ValueError("not find finetune model")
        if (SFT_lora_out_train := finetune_config.get("finetune_lora_model")) is None:
            print("not find SFT_lora_out_train, initial a new lora model")

        if (train_path := finetune_config.get("train_path")) is None:
            if 3 in finetune_steps:
                train_path=pairs_path
                print("not find train file,use pairs_path:{}".format(pairs_path))
            else:
                raise("not find train file")
        
        if (save_dir := finetune_config.get("save_dir")) is None:
            save_dir="lora_out/SFT_out_v2"
            print("not find save_dir, use lora_out/SFT_out_v2")

        if (train_args:=finetune_config.get("train_args")) is None:
            print("not find train_args, use default train_args")
            train_args = TrainingArguments(
                report_to="wandb",
                run_name="DPO_v2",
                output_dir=save_dir,
                per_device_train_batch_size=12,
                gradient_accumulation_steps=2,
                logging_strategy="steps",
                logging_steps=10,
                remove_unused_columns=False,
                num_train_epochs=2,
                save_strategy="steps",
                save_steps=100,
                learning_rate=1e-4,
                save_total_limit=1,
            )
        train(SFT_model,SFT_lora_out_train,train_path,train_args,save_dir=save_dir)

if __name__=="__main__":
    finetune_config={
        "finetune_steps":[4],
        "instruction_dir":"seed_question/machine_generated_instructions.json",
        "generateion_question_lora_model":"lora_out/SFT_out",
        "response_dir":"data/response_result.json",
        "generateion_question_lora_model_2":"lora_out/SFT_out_v2",
        "generateion_question_base_model":"/root/Qwen_7B",
        "generateion_question_base_model_2":"/root/Qwen_7B",
        "score_base_model":"/root/Qwen_7B",
        "score_lora_model":"lora_out/rm_out",
        "finetune_model":"/root/Qwen_7B",
        "train_path":"data/pairs_path.json",
        "finetune_lora_model":"lora_out/SFT_out_v2",
    }
    DPO(finetune_config)