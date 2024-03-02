from transformers import BitsAndBytesConfig, GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from .qwen_generation_utils import make_context, decode_tokens
import random


def inference(instructions, model_name, adapter=None, response_num=1,
              temperature=None, top_p=None,
              max_new_tokens=512, ):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    # model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).eval()

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, quantization_config=bnb_config,
                                                 device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        pad_token='<|extra_0|>',
        padding="left",
        eos_token='<|endoftext|>',
        trust_remote_code=True
    )

    if adapter is not None:
        model.load_adapter(adapter)

    result_instructions = []
    for index in tqdm(range(0, len(instructions), 5)):
        start = index,
        end = min(index + 5, len(instructions))

        batch_raw_text = []
        for j in range(start, end):
            prompt = "Based on content:\n" + instructions[j]["instruction"] + "Answer Question:\n" + instructions[j][
                "input"]
            raw_text, _ = make_context(
                tokenizer,
                prompt,
                system="Now you are a question answering assistant.",
                max_window_size=model.generation_config.max_window_size,
                chat_format=model.generation_config.chat_format,
            )
            batch_raw_text.append(raw_text)

        batch_input_ids = tokenizer(batch_raw_text, padding='longest')
        batch_input_ids = torch.LongTensor(batch_input_ids['input_ids']).to(model.device)
        for _ in range(response_num):
            if temperature is None:
                temperature = random.uniform(0.5, 1)
            if top_p is None:
                top_p = random.uniform(0.05, 0.5)
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_beams=4,
            )
            model.generation_config = generation_config

            batch_out_ids = model.generate(
                batch_input_ids,
                return_dict_in_generate=False,
                generation_config=model.generation_config,
                max_new_tokens=max_new_tokens,
                output_scores=True
            )
            padding_lens = [batch_input_ids[i].eq(tokenizer.pad_token_id).sum().item() for i in
                            range(batch_input_ids.size(0))]
            batch_response = [
                decode_tokens(
                    batch_out_ids[i][padding_lens[i]:],
                    tokenizer,
                    raw_text_len=len(batch_raw_text[i]),
                    context_length=(batch_input_ids[i].size(0) - padding_lens[i]),
                    chat_format="chatml",
                    verbose=False,
                    errors='replace'
                ) for i in range(len(batch_raw_text))
            ]
            for j in range(start, end):
                if hasattr(instructions[j], "output"):
                    result_instruction = {'index': j, 'instruction': instructions[j]["instruction"],
                                          'input': instructions[j]["input"], 'output': instructions[j]["output"],
                                          'generated': batch_response[j - start]}
                else:
                    result_instruction = {'index': j, 'instruction': instructions[j]["instruction"],
                                          'input': instructions[j]["input"], 'generated': batch_response[j - start]}
                result_instructions.append(result_instruction)
    # print("successful generated!")
    return result_instructions
