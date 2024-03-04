import os
import io
import json
import bitsandbytes as bnb



def _make_w_io_base(f, mode: str,is_end=False):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode,encoding="UTF-8")
        if mode=="a":
            if f.tell()!=0:
                if not is_end:
                    f.write(",\n")
            else:
                f.write("[")
    return f

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode,encoding="UTF-8")
    return f

def jdump(obj, f, mode="w", is_end=False,indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode,is_end)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def choose_highest_reward_response(instructions):
    result_instructions={}
    for instruction in instructions:
        if instruction["index"] in result_instructions.keys():
            if result_instructions[instruction["index"]]["score"]<instruction["score"]:
                result_instructions[instruction["index"]]=instruction
        else:
            result_instructions[instruction["index"]]=instruction
    result_list=[]
    for key in result_instructions.keys():
        result_list.append({
            "instruction":result_instructions[key]["instruction"],
            "input":result_instructions[key]["input"],
            "output":result_instructions[key]["generated"],
        })

    return result_list

def combine_pair_response(instructions):
    import math
    result_instructions={}
    for instruction in instructions:
        if instruction["index"] in result_instructions.keys():
            result_instructions[instruction["index"]].append(instruction)
        else:
            result_instructions[instruction["index"]]=[instruction]
    
    pair_instrctions=[]
    for key in result_instructions.keys():
        if len(result_instructions[key])<2:
            continue
        min_index=-1
        min_score=math.inf
        max_index=-1
        max_score=-math.inf
        for i in range(len(result_instructions[key])):
            if result_instructions[key][i]["score"]>max_score:
                max_score=result_instructions[key][i]["score"]
                max_index=i
            if result_instructions[key][i]["score"]<min_score:
                min_score=result_instructions[key][i]["score"]
                min_index=i
        pair_instrctions.append({
            "prompt":"<|im_start|>system\nNow you are a question answering assistant.<|im_end|>\n<|im_start|>user\nBased on content:\n{}Answer Question:\n{}<|im_end|>\n".\
                format(result_instructions[key][0]["instruction"],result_instructions[key][0]["input"]) if "instruction" in result_instructions[key][0].keys() else result_instructions[key][0]["prompt"],
            "chosen":result_instructions[key][max_index]["generated"] if "instruction" in result_instructions[key][0].keys() else result_instructions[key][max_index]["answer"],
            "rejected":result_instructions[key][min_index]["generated"] if "instruction" in result_instructions[key][0].keys() else result_instructions[key][min_index]["answer"],
        })
    return pair_instrctions

def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)