import math
try:
    from utils import jload,jdump
except:
    from utils.utils import jload,jdump
import argparse

def get_argument_data(**kwargs):
    """
    kwargs have args or [k_greedy_path,no_k_greedy_path,save_file_path]
    """
    if "args" in kwargs.keys():
        args=kwargs["args"]
        k_greedy_path=args.k_greedy_path
        no_k_greedy_path=args.no_k_greedy_path
        save_file_path=args.save_file_path
    else:
        try:
            k_greedy_path=kwargs["k_greedy_path"]
            no_k_greedy_path=kwargs["no_k_greedy_path"]
            save_file_path=kwargs["save_file_path"]
        except:
            raise ValueError("dont have [k_greedy_path,no_k_greedy_path,save_file_path] param")
    
    _1 = jload(k_greedy_path)
    _2 = jload(no_k_greedy_path)
    print('number of file 1', len(_1))
    print('number of file 2', len(_2))
    data_list = _1 + _2
    print('number of data', len(data_list))
    jdump(data_list,save_file_path)
    print("success generate argument dataset")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_greedy_path', type=str, 
                        default="argument_data/k_greedy.json", help='Select the k_greedy data')
    parser.add_argument('--no_k_greedy_path', type=str,
                         default="argument_data/no_k_greedy.json", help='Select the no k_greedy data')
    parser.add_argument('--save_file_path', type=str, 
                        default="final_train.json", help='the file to save')
    args, unknown_args = parser.parse_known_args()

    return args






