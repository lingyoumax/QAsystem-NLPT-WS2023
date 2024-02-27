import os
import json
import random
import re
import string
import tqdm
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer
from gpt3_api import make_requests as make_gpt3_requests
from utils import jload,jdump

random.seed(42)


def encode_prompt(prompt_instructions,content_instructions):
    """Encode multiple prompt instructions into a single string."""
    
    prompt = "Reference questions:\n"
    for idx, instruction in enumerate(prompt_instructions):
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt += f"{idx+1}. {instruction}\n"
    prompt += f"Based on the Reference questions.Ask questions about the content:\n"
    prompt+=content_instructions+"\n"
    prompt+="1."
    return prompt

def random_choose_article():
    file=pd.read_csv("seed_question/filter_chunk.csv")
    random_row = file.sample(n=1)
    text=random_row["chunk_text"].tolist()
    text="".join(text[0].split("\n"))
    return text

def sample_machine_instructions(machine_instructions, similarities, n):
    """Sample n machine instructions from a list of machine instructions."""
    return random.sample(machine_instructions, min(n, len(machine_instructions)))


def find_word_in_string(w, s):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)


def post_process_gpt3_response(response):
    if response is None or response["choices"][0]["finish_reason"] == "length":
        return []
    raw_instructions = re.split(r"\n\d+\s?\. ", response["choices"][0]["text"])
    instructions = []
    for inst in raw_instructions:
        inst = re.sub(r"\s+", " ", inst).strip()
        inst = inst.strip().capitalize()
        if inst == "":
            continue
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        if any(find_word_in_string(word, inst) for word in ["image", "images", "graph", "graphs", "picture", "pictures", "file", "files", "map", "maps", "draw", "plot", "go to"]):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result. 
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append(inst)
    return instructions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        default="seed_question/machine_generated",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        default="seed_question/seed_question.json",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--num_instructions_to_generate",
        type=int,
        default=2020,
        help="th",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="gpt-3.5-turbo-instruct",
        help="The engine to use."
    )
    parser.add_argument(
        "--num_prompt_instructions",
        type=int,
        default=8,
        help="The number of instructions to use in the prompt."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send to GPT3 at a time."
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


if __name__ == "__main__":
    args = parse_args()

    #seed_task: question
    seed_tasks=jload(args.seed_tasks_path)

    seed_instructions = [t["Question"] for t in seed_tasks]
    print(f"Loaded {len(seed_instructions)} human-written seed instructions")
    
    os.makedirs(args.batch_dir, exist_ok=True)
    request_idx = 0
    # machine_generated_instructions: question, content
    machine_instructions = []
    if os.path.exists(os.path.join(args.batch_dir, "machine_generated_instructions.json")):
        instruction_info=jload(os.path.join(args.batch_dir, "machine_generated_instructions.json"))
        machine_instructions=[t["Question"] for t in instruction_info]
        request_idx+=len(machine_instructions)
        print(f"Loaded {len(machine_instructions)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    
    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=args.num_instructions_to_generate)
    if machine_instructions:
        progress_bar.update(len(machine_instructions))

    current_index=request_idx
    while len(machine_instructions) < args.num_instructions_to_generate:
        res=[]
        batch_inputs = []
        content_input=[]
        for _ in range(args.request_batch_size):
            prompt_instructions = sample_machine_instructions(
                    machine_instructions, 
                    similarities=None,
                    n=4)
            print(len(prompt_instructions))
            prompt_instructions += random.sample(seed_instructions, args.num_prompt_instructions - len(prompt_instructions))
            random.shuffle(prompt_instructions)
            random_content=random_choose_article()
            prompt = encode_prompt(prompt_instructions,random_content)
            content_input.append(random_content)
            batch_inputs.append(prompt)
        results = make_gpt3_requests(
                engine=args.engine,
                prompts=batch_inputs,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=2,
                stop_sequences=["\n\n", "\n3", "3.", "3."],
                logprobs=1,
                n=1,
                best_of=1,
                api_key=args.api_key,
                organization=args.organization,
            )
        instructions = []
        all_metadata = []
        content=[]
        for c,result in zip(content_input,results):
            new_instructions = post_process_gpt3_response(result["response"])
            instructions += new_instructions
            for i in range(len(new_instructions)):
                content.append(c)
            all_metadata += [result] * len(new_instructions)
        for (c,inst, metadata) in zip(content,instructions, all_metadata):
            # with Pool(4) as p:
            #     rouge_scores = p.map(partial(scorer.score, inst), seed_instructions + machine_instructions)
            rouge_scores =  list(map(partial(scorer.score, inst), seed_instructions + machine_instructions))
            rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]
            if max(rouge_scores) > 0.7:
                    continue
            all_instructions = seed_instructions + machine_instructions
            most_similar_instructions = {
                    all_instructions[i] : rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                }
            machine_instructions.append(inst)
            jdump({
                "Question": inst,
                "Content":c,
                "most_similar": most_similar_instructions,
                "avg_similarity_score": float(np.mean(rouge_scores)),
                "metadata": metadata,
                "request_idx": request_idx
                },
                os.path.join(args.batch_dir, "machine_generated_instructions.json"),
                mode="a")
            progress_bar.update(1)
            request_idx+=1
    if current_index!=request_idx:
        jdump("]",os.path.join(args.batch_dir, "machine_generated_instructions.json"),"a",is_end=True)
        _res=jload(os.path.join(args.batch_dir, "machine_generated_instructions.json"))
        try:
            question_check=_res[0]["Question"]
            res=_res
        except:
            res=[]
            for i in res:
                for j in i:
                    res.append(j)
        jdump(res,os.path.join(args.batch_dir, "machine_generated_instructions.json"))