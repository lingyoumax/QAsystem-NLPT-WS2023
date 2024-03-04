# **1.why need dataset optimization?**
1.The data set we generated directly from chatgpt before was abstract and too granular.
2.Because using the same prompt to generate the data set will lead to the homogeneity of the questions, that is, there are fixed question statements. For example, in hypothetical questions, the sentence pattern "what will happen if xxx" will always be used, which will cause the data set to be unrepresentative.
3.The quality of the generated data set cannot be guaranteed.

# **2.Optimization means**
## I.self instruct[^1] fo QA:
self instruct is an effective method of semi-automatically generating data sets. It cannot then be used directly to generate our dataset, so some modifications are needed to make it suitable for our task.
  - Generation of seed dataset:
      - Because we **don't have enough time and enough perspectives** to generate a good seed data set.Therefore, we use the Question part of the QA data[^2][^3] set on hugging face as the question method we collected from various aspects.
      - Then we use a reward model[^4] to score the question, which essentially just ranks the question.
      - Then randomly sample a part from each level as the seed data to be classified
      - Send these data into the Question_type classification we trained before to get the classification. Select 30 samples from each category as seed data. For the insufficient part, all the data in that category will be used as sub seed dataset, and self instruct (without content) will be executed directly to obtain enough samples.
      - Finally, we obtained 164 pieces of seed data through the above steps.(only have question).
  - Automatically generate data sets:
     - If there is only a seed data set here, 8 will be randomly selected from the seed data set as few shots examples. If there are GPT generated
     - Then the semantic segment obtained through nltk semantic segmentation will be used as the content of the prompt. The specific prompt can be found in templates\clf_task_template.py
     - The remaining part is the same as the original self instruct. Calculate Rouge. If the maximum is greater than 0.7, reject it. Otherwise, accept it as a new Question. So after this step, we have a diverse [Question, Content] data set based on semantic segments.
     - The rest is to answer the question directly, so this part is very simple. The specific prompt can be found in templates\answer_question_template.py.
## II.MoDS[^5]
After the data set is generated, the data set needs to be further filtered and processed. For efficient training, the more representative the training samples are, the better.So we used the MoDS method to conduct further screening.
  - Processing flow
      - Use the reward model[^4] to score each QA pair to obtain a scoring data set.
      - Select a threshold to filter the data set above the threshold.
      - Use a greedy algorithm to filter the encoding representation of the encoded statement, randomly select an initial point, and then select the point farthest from the initial point as the next one, and so on, to obtain the top n furthest seed data sets.
      - Fine-tuning the LLM model[^6] through the seed dataset,then the fine-tuned LLM is used to infer the remaining data sets, and the data sets below the threshold are selected as enhanced data sets.
      - The final data set is obtained by merging the seed data set and the enhanced data set.
# **3.Use in training**
## I. Used in Supervised fine-tuning
Use the final data as a fine-tuning dataset in SFT
## II.Used in Reward model Traing
Here we only use the Question part to allow the model to deduce and obtain the model's answer for further processing to obtain the data set.
## III.Used in DPO
Same as Reward model

## IV.Used in Reject sampling
Same as Reward model

# **Citation**
[^1]:```bibtex
      @misc{selfinstruct,
      title={Self-Instruct: Aligning Language Model with Self Generated Instructions},
      author={Wang, Yizhong and Kordi, Yeganeh and Mishra, Swaroop and Liu, Alisa and Smith, Noah A. and Khashabi, Daniel and Hajishirzi, Hannaneh},
      journal={arXiv preprint arXiv:2212.10560},
      year={2022}
      }
      ```
[^2]:```bibtex
      @article{2016arXiv160605250R,
      author = {{Rajpurkar}, Pranav and {Zhang}, Jian and {Lopyrev},Konstantin and {Liang}, Percy},
      title = "{SQuAD: 100,000+ Questions for Machine Comprehension of Text}",
      journal = {arXiv e-prints},
      year = 2016,
      id = {arXiv:1606.05250},
      pages = {arXiv:1606.05250},
      archivePrefix = {arXiv},
      eprint = {1606.05250},
        }
      }
      ```
[^3]:```bibtex
      @online{DatabricksBlog2023DollyV2,
      author    = {Mike Conover and Matt Hayes and Ankit Mathur and Jianwei Xie and Jun Wan and Sam Shah and Ali Ghodsi and Patrick Wendell and Matei Zaharia and Reynold Xin},
      title     = {Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM},
      year      = {2023},
      url       = {https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm},
      urldate   = {2023-06-30}
      }
      ```
[^4]:https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2
[^5]:```bibtex
      @misc{du2023mods,
      title={MoDS: Model-oriented Data Selection for Instruction Tuning}, 
      author={Qianlong Du and Chengqing Zong and Jiajun Zhang},
      year={2023},
      eprint={2311.15653},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
    }
    ```
[^6]:```bibtex
    @article{qwen,
    title={Qwen Technical Report},
    author={Jinze Bai and Shuai Bai and Yunfei Chu and Zeyu Cui and Kai Dang and Xiaodong Deng and Yang Fan and Wenbin Ge and Yu Han and Fei Huang and Binyuan Hui and Luo Ji and Mei Li and Junyang Lin and Runji Lin and Dayiheng Liu and Gao Liu and Chengqiang Lu and Keming Lu and Jianxin Ma and Rui Men and Xingzhang Ren and Xuancheng Ren and Chuanqi Tan and Sinan Tan and Jianhong Tu and Peng Wang and Shijie Wang and Wei Wang and Shengguang Wu and Benfeng Xu and Jin Xu and An Yang and Hao Yang and Jian Yang and Shusheng Yang and Yang Yao and Bowen Yu and Hongyi Yuan and Zheng Yuan and Jianwei Zhang and Xingxuan Zhang and Yichang Zhang and Zhenru Zhang and Chang Zhou and Jingren Zhou and Xiaohuan Zhou and Tianhang Zhu},
    journal={arXiv preprint arXiv:2309.16609},
    year={2023}
    }
    ```
