import math
try:
    from utils import jload,jdump
except:
    from utils.utils import jload,jdump
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.0, help='Selection of results above the threshold')
    parser.add_argument('--score_data_path', type=str,
                         default="argument_data/score_data.json", help='Select the file to be judged')
    parser.add_argument('--save_file_path', type=str, default="argument_data/high_quality.json", help='Select the file to be judged')
    args, unknown_args = parser.parse_known_args()

    return args

def get_high_quality_result(**kwargs):
    """
    args: have threshold score_data_path save_file_path
    if want to use function call, args should be None, and use kwargs to get parameters
    """
    if "args" in kwargs.keys():
        args=kwargs["args"]
        threshold = args.threshold
        score_data_path=args.score_data_path
        high_quality_file=args.save_file_path
    else:
        try:
            threshold=kwargs["threshold"]
            score_data_path=kwargs["score_data_path"]
            save_file_path=kwargs["save_file_path"]
        except:
            raise ValueError("dont have threshold,score_data_path,save_file_path parameters")
    
    score_data = jload(score_data_path)
    all_num = len(score_data)
    print('all number of instructions', len(score_data))
    num_dict = {}
    result_json = []
    for item in score_data:
        upper_num = math.ceil(item['reward_score'])
        lower_num = math.floor(item['reward_score'])
        num_dict[(lower_num, upper_num)] = num_dict.get((lower_num,upper_num),0) + 1
        if float(item['reward_score']) > threshold:
            result_json.append(item)

    print('The percent of each score interval:')
    for k, v in num_dict.items():
        print(str(k)+'  :  '+str(v)+'  '+str(float(v)/all_num))

    print('num of good case : ',len(result_json))

    #jdump(result_json,result_file)
    jdump(result_json,high_quality_file)
    print("successfull high quality choose!")

if __name__=="__main__":
    args=get_args()
    get_high_quality_result(args)

    
    
    
    
