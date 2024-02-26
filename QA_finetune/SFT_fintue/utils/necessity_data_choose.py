import argparse
try:
    from utils.utils import jload,jdump
except:
    from utils import jload,jdump
import math

def necessity_data_choose(**kwargs):
    """
    kwargs have args or [threshold,scored_path,save_file_path]
    """
    if "args" in kwargs.keys():
        args=kwargs["args"]
        threshold=args.threshold
        scored_path=args.scored_path
        save_file_path=args.save_file_path
    else:
        try:
            threshold=kwargs["threshold"]
            scored_path=kwargs["scored_path"]
            save_file_path=kwargs["save_file_path"]
        except:
            raise ValueError("dont have [threshold,scored_path,save_file_path] parameters")
    
    score_list = jload(scored_path)
    all_num = len(score_list)
    print('all number of instructions', len(score_list))

    num_dict = {}
    result_json = []
    for item in score_list:
        upper_num = math.ceil(item['reward_score'])
        lower_num = math.floor(item['reward_score'])
        num_dict[(lower_num, upper_num)] = num_dict.get((lower_num,upper_num),0) + 1
        del item["generated"]
        if float(item['reward_score']) < threshold:
            result_json.append(item)

    print('The percent of each score:')
    for k, v in num_dict.items():
        print(str(k)+'  :  '+str(v)+'  '+str(float(v)/all_num))

    print('num of bad case : ',len(result_json))
    jdump(result_json,save_file_path)
    print("sucess choose data set")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scored_path', type=str, default="argument_data/no_k_greedy_generate_score.json", help='choose no greedy scored dataset')
    parser.add_argument('--threshold', type=float,
                         default=0.0, help='reward model name')
    parser.add_argument('--save_file_path', type=str, default="argument_data/no_k_greedy_choosen.json", help='save chosen no greedy score dataset')
    args, unknown_args = parser.parse_known_args()

    return args