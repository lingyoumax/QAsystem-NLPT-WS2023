import os
import json
import random
import tqdm
import re
import argparse
import pandas as pd
from collections import OrderedDict
from gpt3_api import make_requests as make_gpt3_requests
from templates.answer_question_template import prompt_qa
from utils import jdump,jload

random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        default="seed_question/machine_generated",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="machine_generated_instructions.json"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="machine_generated_instances.json",
    )
    parser.add_argument(
        "--max_instances_to_generate",
        type=int,
        default=2,
        help="The max number of instances to generate for each instruction.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="gpt-3.5-turbo-instruct",
        help="The engine to use."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send in a batch."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="The API key to use. If not specified, the key will be read from the environment variable OPENAI_API_KEY."
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="The organization to use. If not specified, the default organization id will be used."
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    lines=jload(os.path.join(args.batch_dir, args.input_file))
    for i in range(len(lines)):
        if "metadata" in lines[i]:
            lines[i]["instruction_metadata"] = lines[i]["metadata"]
            del lines[i]["metadata"]

    output_path = os.path.join(args.batch_dir, args.output_file)

    progress_bar = tqdm.tqdm(total=len(lines))
    for batch_idx in range(0, len(lines), args.request_batch_size):
        batch = lines[batch_idx: batch_idx + args.request_batch_size]
        prompts = []
        for task in batch:
            prompt=prompt_qa+task["Question"]+"\n"+"Content:"+task["Content"]+"\n"+"Answer:"
            prompts.append(prompt)
        results = make_gpt3_requests(
            engine=args.engine,
            prompts=prompts,
            # because the clf template is longer, we need to decrease the max_tokens
            max_tokens=1024,
            temperature=0,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=1.5,
            stop_sequences=["Content:", "Question:"],
            logprobs=1,
            n=1,
            best_of=1,
            api_key=args.api_key,
            organization=args.organization)
        for i in range(len(batch)):
            data = batch[i]
            
            data["instance_metadata"] = results[i]
            if results[i]["response"] is not None:
                data["raw_instances"] = results[i]["response"]["choices"][0]["text"]
            else:
                data["raw_instances"] = ""
            jdump({"input":batch[i]["Question"],
                   "instruction":batch[i]["Content"],
                   "output":data["raw_instances"]},output_path,"a")
        progress_bar.update(len(batch))
    jdump("]",output_path,"a",is_end=True)
    _res=jload(output_path)
    try:
        question_check=_res[0]["Question"]
        res=_res
    except:
        res=[]
        for i in res:
            for j in i:
                res.append(j)
    jdump(res,output_path)