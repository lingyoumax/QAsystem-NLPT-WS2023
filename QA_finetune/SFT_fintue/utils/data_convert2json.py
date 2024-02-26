import pandas as pd
import json
import argparse

def convert2json(**kwargs):
    """
    base_data_path: should be [PMID,PubDate,Authors,Abstract,Keywords,ArticleTitle], at least have PMID and Abstract
    convert_data_path: should be [Question,Answer,PMID]
    save_file_path: the result save path
    """
    if "args" in kwargs.keys():
        args=kwargs["args"]
        base_data_path=args.base_data_path
        convert_data_path=args.convert_data_path
        save_file_path=args.save_file_path
    else:
        try:
            base_data_path=kwargs["base_data_path"]
            convert_data_path=kwargs["convert_data_path"]
            save_file_path=kwargs["save_file_path"]
        except:
            raise ValueError("dont have [base_data_path,convert_data_path,save_file_path] param")
    try:
        df_a = pd.read_csv(base_data_path)
        df_b = pd.read_csv(convert_data_path)
    except:
        raise FileNotFoundError

    if "PMID" not in df_a.columns or "Abstract" not in df_a.columns:
        raise ValueError("The 'base_data_path' DataFrame must have [PMID,Abstract] columns.")

    if 'PMID' not in df_b.columns or 'Question' not in df_b.columns or "Answer" not in df_b.columns:
        raise ValueError("The 'convert_data_path' DataFrame must have [PMID,Question,Answer] columns.")

    merged_df = pd.merge(df_b, df_a[['PMID', 'Abstract']], on='PMID', how='left')

    final_df = merged_df[['Question', 'Answer', 'Abstract']]
    final_df.columns = ['input', 'output', 'instruction']
    final_df=final_df.dropna()
    data_list = final_df.to_dict('records')

    if "json" not in save_file_path:
        raise ValueError("The 'save_file_path' should be json format.")
    with open(save_file_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
    print("sucessful merge and convert!")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_data_path', type=str, default="raw_data/PubmedDataSet.csv", help='Selection of the base dataset')
    parser.add_argument('--convert_data_path', type=str,
                         default="raw_data/train_21(1).csv", help='sele the qa dataset')
    parser.add_argument('--save_file_path', type=str, default="convert_data/merged_dataset.json", help='Select the file to be saved')
    args, unknown_args = parser.parse_known_args()

    return args

if __name__=="__main__":
    base_data_path="raw_data/PubmedDataSet.csv"
    convert_data_path="raw_data/train_21(1).csv"
    save_file_path="convert_data/merged_dataset.json"
    convert2json(base_data_path=base_data_path,convert_data_path=convert_data_path,save_file_path=save_file_path)