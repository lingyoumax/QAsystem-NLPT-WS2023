from utils.data_convert2json import convert2json
from utils.quality_evaluation import exact_score
from utils.high_quality_exaction import get_high_quality_result
from utils.k_greedy import k_greedy
from utils.get_not_in_greedy_data import get_not_in_greedy_data
from utils.necessity_eval import necessity_eval
from utils.necessity_data_choose import necessity_data_choose
from utils.necessity_eval import necessity_eval
from utils.get_argument_data import get_argument_data

convert2json("raw_data/PubmedDataSet.csv",
                 "raw_data/train_21(1).csv",
                 "convert_data/merged_dataset.json")
exact_score(model_name="OpenAssistant/reward-model-deberta-v3-large-v2",
            quality_evaluation_file="convert_data/merged_dataset.json",
            save_file_path="argument/high_quality.json")
get_high_quality_result(threshold=0.0,
                        score_data_path="argument_data/score_data.json",
                        save_file_path="argument_data/high_quality.json")
k_greedy(high_quality_data_path="argument_data/high_quality.json",
         k_greedy_save_path="argument_data/k_greedy.json",
         top_k=2000)
get_not_in_greedy_data(high_quality_json="argument_data/high_quality.json",
                        k_greedy_json="argument_data/k_greedy.json",
                          divided_num=5000,
                          save_file_name="argument_data/no_k_greedy.json")
# now we have the greedy dataset 
# and use this to fintune model
# and after train mode, inference the data not in greedy dataset but in high quality dataset
# assume we have inference no_k_greedy generate data in "SFT_dataset/argument_data/no_k_greedy_generate.json"

no_k_greedy_path="argument_data/no_k_greedy_generate.json"
model_name="OpenAssistant/reward-model-deberta-v3-large-v2"
save_path="argument_data/no_k_greedy_generate_score.json"
necessity_eval(no_k_greedy_path=no_k_greedy_path,model_name=model_name,save_path=save_path)

scored_path="argument_data/no_k_greedy_generate_score.json"
threshold=0.0
save_path="argument_data/no_k_greedy_choosen.json"
necessity_data_choose(scored_path=scored_path,threshold=threshold,save_path=save_path)

no_k_greedy_path="argument_data/no_k_greedy_choosen.json"
model_name="reward-model-deberta-v3-large-v2"
save_file_path="argument_data/no_k_greedy_score.json"
necessity_eval(no_k_greedy_path=no_k_greedy_path,model_name=model_name,save_file_path=save_file_path)

k_greedy_path="argument_data/k_greedy.json"
no_k_greedy_path="argument_data/no_k_greedy_score.json"
save_file_path="final_train_data.json"
get_argument_data(k_greedy_path=k_greedy_path,no_k_greedy_path=no_k_greedy_path,save_file_path=save_file_path)