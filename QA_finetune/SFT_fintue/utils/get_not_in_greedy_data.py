import math
try:
    from utils import jload,jdump
except:
    from utils.utils import jload,jdump
import argparse

def get_not_in_greedy_data(**kwargs):
    """
    args: have high_quality_json k_greedy_json divided_num: the num of instances in each saved json file save_file_path
    if want to use function call, args should be None, and use kwargs to get parameters
    save_file_path,dont have json end
    """
    if "args" in kwargs.keys():
        args=kwargs["args"]
        high_quality_json=args.high_quality_json
        k_greedy_json=args.k_greedy_json
        divided_num=args.divided_num
        save_file_path=args.save_file_path
    else:
        try:
            high_quality_json=kwargs["high_quality_json"]
            k_greedy_json=kwargs["k_greedy_json"]
            divided_num=kwargs["divided_num"]
            save_file_path=kwargs["save_file_path"]
        except:
            raise ValueError("dont have [high_quality_json,k_greedy_json,divided_num,save_file_path] param")
    high_quality_json=jload(high_quality_json)
    print(len(high_quality_json))
    k_greedy_json=jload(k_greedy_json)
    print(len(k_greedy_json))
    k_greedy_json_set = set((item['input'],item['instruction']) for item in k_greedy_json)
    k_greedy_json_set_list=list(k_greedy_json_set)
    print(len(k_greedy_json_set_list))
    delete_list=[]
    for i in k_greedy_json_set_list:
        for j in range(len(high_quality_json)):
            input=high_quality_json[j]["input"]
            instuction=high_quality_json[j]["instruction"]
            if i[0]==input and i[1]==instuction:
                delete_list.append(j)
    print(len(delete_list))
    delete_list=sorted(delete_list,reverse=True)
    for i in delete_list:
        del high_quality_json[i]
    print(len(high_quality_json))
    if divided_num>=len(high_quality_json):
        jdump(high_quality_json,"{}.json".format(save_file_path))
    else:
        for i in range(0,len(high_quality_json),divided_num):
            start=i
            end=min(i+divided_num,len(high_quality_json))
            jdump(high_quality_json[start:end],"{}_{}.json".format(save_file_path,i//divided_num))
    print("success get not in greedy dataset")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--high_quality_json', type=str, 
                        default="argument_data/high_quality.json", help='Selection of high quality data')
    parser.add_argument('--k_greedy_json', type=str,
                         default="argument_data/k_greedy_data.json", help='Select the k_greedy data')
    parser.add_argument('--divided_num', type=int, default=800, help='the number of instances in each saved json')
    parser.add_argument('--save_file_path', type=str, 
                        default="argument_data/no_greedy_data", help='the file to save,dont have json end')
    args, unknown_args = parser.parse_known_args()

    return args


