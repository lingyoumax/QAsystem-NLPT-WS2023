# QA fintuen using Qwen-7B-Chat[^1]
## 1. Model Processing Flow
### I. Data set processing
- a.Generation of more data sets：Use the self instuct[^2] method to generate the data set using the QA seed library we concocted.But in contrast to the generation for generic domains used in the original approach, we set the INSTRUCTION to the relevant statement segment, the INPUT to the different types of questions, and the OUTPUT to the corresponding answers. Similarly it is also categorized into judgmental and non-judgmental questions.
- b.Using the MoDS[^3] method to filter out higher quality datasets.
### II. supervised fine-tuning
- a.  We used the enhanced dataset obtained earlier to accomplish this step of fine-tuning.
- b.Quantization at model load due to resource issues
- c.Compared to full fine-tuning, we use the adapter[^4] approach which is less memory intensive and more efficient.
### III.Reward model
- Waiting for completion
### IV. PPO 
- Waiting for completion

## 2.Environment Configuration
- a.Training configuration
  - random access memory (RAM):Minimum 15GB
  - Display card：4090,24GB(for training, at least 15GB)
- b.Inference Configuration
  -  random access memory (RAM):Minimum 5GB
  -  Display card：4090,24GB(for inference, at least 12GB, if for batch inference, Recommended minimum 18GB)
## 3.Deployment of models
- Waiting for completion








## **Citation**:
[^1]:```bibtex
    @article{qwen,
    title={Qwen Technical Report},
    author={Jinze Bai and Shuai Bai and Yunfei Chu and Zeyu Cui and Kai Dang and Xiaodong Deng and Yang Fan and Wenbin Ge and Yu Han and Fei Huang and Binyuan Hui and Luo Ji and Mei Li and Junyang Lin and Runji Lin and Dayiheng Liu and Gao Liu and Chengqiang Lu and Keming Lu and Jianxin Ma and Rui Men and Xingzhang Ren and Xuancheng Ren and Chuanqi Tan and Sinan Tan and Jianhong Tu and Peng Wang and Shijie Wang and Wei Wang and Shengguang Wu and Benfeng Xu and Jin Xu and An Yang and Hao Yang and Jian Yang and Shusheng Yang and Yang Yao and Bowen Yu and Hongyi Yuan and Zheng Yuan and Jianwei Zhang and Xingxuan Zhang and Yichang Zhang and Zhenru Zhang and Chang Zhou and Jingren Zhou and Xiaohuan Zhou and Tianhang Zhu},
    journal={arXiv preprint arXiv:2309.16609},
    year={2023}
    }
    ```
[^2]:```bibtex
      @misc{selfinstruct,
      title={Self-Instruct: Aligning Language Model with Self Generated Instructions},
      author={Wang, Yizhong and Kordi, Yeganeh and Mishra, Swaroop and Liu, Alisa and Smith, Noah A. and Khashabi, Daniel and Hajishirzi, Hannaneh},
      journal={arXiv preprint arXiv:2212.10560},
      year={2022}
      }
      ```
[^3]:```bibtex
      @misc{du2023mods,
      title={MoDS: Model-oriented Data Selection for Instruction Tuning}, 
      author={Qianlong Du and Chengqing Zong and Jiajun Zhang},
      year={2023},
      eprint={2311.15653},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
    }
    ```
[^4]:```bibtex
      @Misc{peft,
      title =        {PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods},
      author =       {Sourab Mangrulkar and Sylvain Gugger and Lysandre Debut and Younes Belkada and Sayak Paul and Benjamin Bossan},
      howpublished = {\url{https://github.com/huggingface/peft}},
      year =         {2022}
      }
     ```
