import json
def getData(file_path = 'dataset/dataset_SFTModel.json'):
    
    # 以读取模式打开文件
    with open(file_path, 'r') as file:
        # 使用json.load()读取文件内容
        data = json.load(file)
    return data
