# QAsystem-NLPT

### Natural Language Processing with Transformers Project

##### **Key Information**

Title: QAsystem-NLPT

**Team Members:**

Guangdeng Liang, 3769325, Data and computer science (GitHub: Gorden-GitHub)

Ye Tao, 4734192, Data and computer science (Github: lingyoumax)

Yong Wu, 3770613，Data and computer science (Github:yongwu_cs)

Ziwei Liu, 3766789, Data and computer science (GitHub: ZiweiLiu0908)

**Mail Addresses:**

[guangdeng.liang@stud.uni-heidelberg.de](mailto:guangdeng.liang@stud.uni-heidelberg.de)

ye.tao@stud.uni-heidelberg.de 

yong.wu@stud.uni-heidelberg.de

ZiweiLiu0908@gmail.com

**Member Contribution:** - 

###### Guangdeng Liang

Text_retrieval, Data preprocessing, Retrieval evaluation, Backend, Integration and Docker Deployment 

###### Ye Tao   

Dataset generation，Model evaluation, Dataset evaluation - 

###### Yong Wu  

Dataset generation, Dataset augmentation, QA finetune 

###### Ziwei Liu 

Data Acquisition, Text_retrieval, Backend, Frontend, Retrieval evaluation

**Advisor:**

Satya Almasian



**Anti-plagiarism Confirmation:**







## Introduction

Our project presents an innovative Retrieval-Augmented Generation (RAG)[^1] System that improves information retrieval and answer generation by utilizing advanced text analysis techniques. This system is crucial for handling complex queries that require understanding context and nuances to produce accurate and relevant responses. It addresses the growing need for sophisticated systems that can navigate the complexities of human language and vast data repositories. Our method integrates semantic and lexicographical search strategies for thorough and precise data retrieval. This is followed by creating a concise reference summary, which forms the basis for generating coherent and informative answers.

We use a BM25-based[^2] algorithm for lexicographic search to ensure relevance to the query terms and a fine-tuned Large Language Model for semantic search to capture the user's intent. The Pseudo-Relevance Feedback (PRF) method refines search results by evaluating their relevance.

To address the challenges of non-standard training data and limited resources in natural language processing, we introduce a semi-automatic dataset optimization approach. This approach combines the self-instruct[^3] method for generating diverse question-answer pairs and the Model-oriented Data Selection for Instruction Tuning (MoDS)[^4] method for refining the dataset by focusing on quality, coverage, and necessity. Utilizing GPT-3.5 for data annotation has significantly enhanced efficiency and quality[^5]. For model fine-tuning, we employed QLoRA technology for efficient model training and memory reduction[^6], starting with the Qwen-7B-Chat model[^7]. Our methodology includes a step-by-step training strategy, starting with Supervised Fine-Tuning (SFT)[^8], followed by iterative fine-tuning using rejection sampling and Direct Preference Optimization (DPO) for continuous improvement[^9]. This process aims to enhance the model's performance and generalization on specific NLP tasks.

The report is structured to give a thorough overview of our RAG System, including a review of related work, our methodologies, experimental setup, evaluation methods, results, and analysis. We conclude with our contributions to the field and future research directions for retrieval-augmented systems.



## **Related Work**

The landscape of information retrieval and answer generation has been significantly shaped by various pioneering works, each contributing to the evolution of methodologies and systems tailored for enhanced understanding and processing of natural language queries. Our approach draws inspiration from and seeks to advance beyond the foundational principles laid out in these seminal studies.

A core component of our retrieval system is based on the well-established BM25 algorithm, a probabilistic information retrieval model that has been widely recognized for its effectiveness in ranking documents based on the occurrences of query terms within them. We extend this foundation by integrating the Pseudo-Relevance Feedback (PRF) method, a technique designed to refine initial search results by assuming the top-ranked documents are relevant and using their content to modify the original query, thereby improving retrieval accuracy[^10]. In parallel, our system incorporates semantic search capabilities, leveraging advancements in Large Language Models (LLMs) to understand the contextual nuances of queries [^11] . This dual approach of combining lexicographical precision with semantic comprehension aligns with recent trends in retrieval-augmented generation systems, which emphasize the importance of capturing both explicit and implicit query intents[^12].

In the domain of natural language processing, the quality of training datasets plays a pivotal role in enhancing model performance[^13]. Traditionally, dataset construction has heavily relied on labor-intensive manual annotation, which constrains the diversity and volume of available data. Recently, semi-automated approaches like self instruct have emerged, offering a promising solution to these challenges. This method leverages high quality prompts to generate higher-quality data and employs Rouge metrics to filter out excessively repetitive content, facilitating the creation of expansive, high-quality datasets[^3]. Moreover, the MoDS method introduces a nuanced approach to dataset refinement by evaluating potential data based on three criteria: quality, coverage, and necessity. Utilizing a combination of a greedy algorithm and a necessity detection algorithm, MoDS effectively identifies and selects the most valuable data for training from a vast pool of candidates[^4].

To address the constraints of resources and time in model fine-tuning, QLoRA technology emerges as an effective solution. Specifically, the QLoRA approach has demonstrated superior performance on small, high-quality datasets by performing low-rank decomposition of linear layers in large models to simulate parameter updates, thereby optimizing the fine-tuning process[^6].
OpenAI's standardized process for large model fine-tuning, which encompasses supervised fine-tuning, reward model fine-tuning, and the application of the Proximal Policy Optimization (PPO) strategy, has yielded significant advancements[^14]. However, unlike the extensive requirements of PPO[^16], which includes training multiple models for reinforcement learning, fine-tuning with language model sampling, and extensive hyperparameter tuning, the Direct Policy Optimization (DPO) method simplifies this process. DPO eliminates the need for additional reward model training and aligns more closely with human preferences without requiring substantial adjustments.

What distinguishes our work is the innovative application of a hierarchical query mechanism that first amplifies the importance of key terms in the query and then processes it through the BM25 algorithm. This adaptation, inspired by work on keyword amplification in query reconstruction [^15], significantly improves the accuracy of the retrieval phase. Additionally, our unique integration of PRF with semantic and lexicon search strategies in a new system architecture further improves the accuracy of retrieved information and addresses limitations observed in previous systems where either strategy was applied alone. In addition, the process of self instruct has been modified to make it more suitable for our closed QA tasks. And the rejection sampling method of llama2 is introduced into the openAI training process and DPO is used instead of PPO as the last step of reinforcement learning training, so that individuals can complete the entire training process.

## **Methods/Approach** 

### **Text Retrieval:**

#### Semantic Search: 

##### Chunking Methods

For effective processing and analysis of text data within our project, implementing an efficient chunking method is essential. One significant consideration is that some embedding models and answer generation models impose restrictions on the maximum number of input tokens they can handle. This limitation necessitates the division of text into smaller, manageable pieces or "chunks" to ensure compatibility with these models' input constraints. Chunking, therefore, becomes not just a matter of processing efficiency but also a fundamental requirement for the operational feasibility of our models. 

To address this, we have opted to utilize the [NLTK Text Splitte](https://api.python.langchain.com/en/latest/text_splitter/langchain.text_splitter.NLTKTextSplitter.html), chosen for its robustness and its ability to accurately segment text based on natural language cues, ensuring that each chunk adheres to the models' input token limits while maintaining coherent and contextually meaningful segments of text. This splitter leverages the [Natural Language Toolkit](https://www.nltk.org/), a leading platform for building Python programs to work with human language data, which provides powerful linguistic processing capabilities.

##### Embedding Model

In our pursuit of a highly efficient and effective embedding model for our text processing pipeline, we carefully reviewed various options presented in the [rag.pdf](https://moodle.uni-heidelberg.de/pluginfile.php/1371515/mod_resource/content/2/rag.pdf), particularly focusing on the recommendations provided on [Hugging Face's MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard). After thorough consideration, we decided to adopt the [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) model as our primary embedding model.
Our selection of the bge-large-en-v1.5 model as the embedding model for our project was principally guided by its high ranking on Hugging Face's MTEB leaderboard. The leaderboard showcases a comprehensive evaluation of various models based on their performance across multiple text embedding benchmarks. The bge-large-en-v1.5 model's superior position indicates its exceptional capability in generating robust and versatile embeddings, making it a highly suitable choice for our project's intricate requirements in text retrieval and analysis.

Given the specific nature of our project's domain, fine-tuning the bge-large-en-v1.5 model on our dataset is a crucial step. This process involves adjusting the model's parameters to better align with the semantic and contextual nuances of our collected text data, particularly the abstracts sourced from PubMed that contain the keyword "intelligence." Fine-tuning ensures that the model's embeddings are highly relevant and optimized for our text corpus, thereby enhancing the accuracy and relevance of the text retrieval and answer generation components of our system.

##### Vector Database

For the storage, management, and retrieval of embedding vectors generated by the bge-large-en-v1.5 model, our team initially chose [OpenSearch](https://opensearch.org/) as our vector database solution. OpenSearch, a community-driven, open-source search and analytics suite, offers robust capabilities for handling large-scale vector data, making it an ideal choice for our project's needs. We successfully implemented the storage and querying functionalities using OpenSearch and deployed it through Docker for ease of use.
However, as our project progressed, we encountered several challenges with this setup. Our team members found it inconvenient to set up and replicate the Docker environment for testing purposes. Additionally, we faced difficulties in transferring the vector data stored in OpenSearch to other team members for further analysis and testing.

Given these challenges, we decided to transition to using [Pinecone](https://www.pinecone.io/) for our vector database needs. Pinecone is designed specifically for vector search and provides a more user-friendly and efficient way to manage and share vector data among team members, addressing the issues we faced with OpenSearch and Docker. This switch has significantly improved our workflow and collaboration, enabling us to focus on developing and refining our project further.

#### Lexicographical Search:

##### Baseline Approach BM25:

The BM25 algorithm is a widely-used ranking function in text retrieval, designed to evaluate the relevance of documents to a search query. It improves upon the simpler [TF-IDF]([https://en.wikipedia.org/wiki/Tf%E2%80%93idf](https://en.wikipedia.org/wiki/Tf–idf)) model by considering:Term Frequency (TF): How often a query term appears in a document, with diminishing returns for higher frequency to prevent overemphasis on term count.Document Length: Adjusts scores to prevent bias towards longer documents, ensuring fair comparison across documents of varying lengths.Inverse Document Frequency (IDF): Gauges the importance of a term based on its rarity across all documents, giving higher weight to rarer terms.The BM25 score for a document D given a query Q with terms q1,q2,...,qn is calculated as:
$$
Score(D, Q) = \sum_{i=1}^{n} IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{avgdl})}
$$


​				
​					

Where: f(q_i, D) is q_i's term frequency in the document D,

|D| is the length of the document D,

avgdl is the average document length in the text collection,

IDF(q_i) is the inverse document frequency of the term q_i,

k_1 and b are free parameters, usually chosen, in absence of advanced optimization, as k_1 ∈ [1.2, 2.0] and b = 0.75.

##### **Our Approach:**

###### Intitution:

Our method, inspired by Weighted BM25, assigns different weights to different words in a query, rather than to assign different weights to defferent document fragments. We prioritize keywords within our query by assigning higher weights to them, ensuring BM25 focuses more on these essential keywords. For instance, in the question "What were the survival rates for infants born between 22 and 24 weeks of gestation in Western Australia, and what were the reported neurodevelopmental outcomes at ages 3-5 years for those infants who survived?" keywords such as 'infants', 'weeks', 'survived', 'Western', 'Australia', 'gestation', and 'born' are given greater significance. This approach enhances the relevancy of search results by emphasizing the most critical aspects of the query.

We use yake, a prowerful tool to extract keywords in paragraph, to extract keywords from the query.

```python
import yake
kw_extractor = yake.KeywordExtractor(n=1, dedupLim=0.9, top=10, features=None)
```

Then we try to increase the weight of those keywords in BM25 algorithm. A intutive and useful method is to duplicate the keywords in the query. For example, a query 

`['What', 'were', 'the', 'survival', 'rates', 'for', 'infants', 'born', 'between', '22', 'and', '24', 'weeks', 'of', 'gestation', 'in', 'Western', 'Australia,', 'and', 'what', 'were', 'the', 'reported', 'neurodevelopmental', 'outcomes', 'at', 'ages', '3-5', 'years', 'in', 'those', 'infants', 'who', 'survived?']`

After we duplicate the keywords in it, we have

 `['What', 'were', 'the', 'survival', 'rates', 'for', 'infants', 'born', 'between', '22', 'and', '24', 'weeks', 'of', 'gestation', 'in', 'Western', 'Australia,', 'and', 'what', 'were', 'the', 'reported', 'neurodevelopmental', 'outcomes', 'at', 'ages', '3-5', 'years', 'in', 'those', 'infants', 'who', 'survived?', 'Australia', 'reported', 'infants', 'survived', 'survival', 'weeks', 'rates', 'gestation', 'Western', 'born']` 

Then we send this query-keywords list to BM25 algorithm to get the score for each document.

###### Mathematical explaination:

Increasing the value of f(qi,D) will increase the numerator f(qi,D)⋅(k1+1) since k1is a constant. However, the denominator will also increase because it contains the term f(qi,D). But in the denominator, the term frequency f(qi,D) is moderated by the constant k1, as well as the normalization part related to the document length ∣D∣ and the parameter b, which means that the rate of increase in the denominator is not as steep as that of the numerator.

Thus, as f(qi,D) increases, the overall score increases, but due to the denominator, this increase is subject to diminishing returns, a phenomenon known as term frequency saturation. After a certain point, further increases in term frequency have progressively smaller contributions to the score. This prevents the situation where a term's frequency excessively amplifies the document's relevance due to its repetitive occurrence. 

###### Hierarchical Search:

Utilizing this enhanced BM25 algorithm on each text chunk could significantly increase the time cost. Therefore, we initially apply the algorithm to the entire abstracts, selecting the top 30 based on their relevance. Subsequently, we employ the algorithm on the text chunks within these top 30 abstracts to refine our results. This two-step approach balances efficiency with precision, ensuring a thorough yet time-effective search process.

###### Combined PRF Search:

In our system, the integration of semantic search and lexicographical search is enhanced by the Pseudo-Relevance Feedback (PRF) method. PRF operates by initially using the input question to perform a preliminary search, yielding a set of reference abstracts deemed relevant. Subsequently, these abstracts are treated as 'pseudo-queries'. The core of PRF is the assumption that the top results from the initial retrieval are relevant, and it uses information from these results to refine the search process. By treating the top documents or terms from these initial results as a new query, the system searches the database once more to retrieve additional relevant documents. This iterative process serves to bridge the gap between the literal terms in the user's query and the conceptual topics within the database, thus enhancing the overall relevance of the retrieved documents.
In our system, after the initial search we also combine the query with the first-stage references as the input of the second-stage research. 

### **Answer Generation:**

#### **Dataset optimization methods**

##### **Self instruct**

In order to improve the quality, diversity and creativity of the data set, we adopted an improved self instruct method to generate questions. The traditional method is through manual annotation, which is time-consuming and slow in efficiency. In addition, we can ask questions about GPT by giving the same prompt, but this will cause the consistency of the questions to be too high, and the query granularity for a single abstract will be too large, failing to fully tap the information potential of each abstract. For example, when generating hypothetical questions (such as "what would happen if" type questions), we found that asking questions using only fixed templates often failed to generate sufficiently diverse questions.

To solve this problem, we simplified the original self instruct process and adjusted it into two main steps, which is more suitable for our closed QA task. First, we use a seed question and part of a previously generated question to build a question template to generate new questions. Then, use these questions and corresponding content to generate answers. This approach not only enhances the diversity and creativity of the questions, but also improves the quality of the entire data set. In this way, we can generate richer and deeper question-answer pairs and better utilize the information provided by each abstract, thus providing a high-quality data set for subsequent question and answer tasks.

Specifically, it includes the following steps:

1. Construct the seed data setConsidering that the quality of question generation directly affects the effectiveness of the entire dataset, we first set out to construct a high-quality seed dataset. To this end, we use two QA datasets released on hugging face: [SQuAD2]([The Stanford Question Answering Dataset (rajpurkar.github.io)](https://rajpurkar.github.io/SQuAD-explorer/)) and [databricks-dolly]([databricks/databricks-dolly-15k · Datasets at Hugging Face](https://huggingface.co/datasets/databricks/databricks-dolly-15k)) as the basis. By using the [bert-base-uncased]([google-bert/bert-base-uncased · Hugging Face](https://huggingface.co/google-bert/bert-base-uncased)) model to encode the questions in these datasets into sentence embeddings, we selected the top 2000 questions with the farthest sentence embeddings as the preliminary screening results. Subsequently, we used our [self-trained question type classifier]([QAsystem-NLPT-WS2023/Question_type at main · lingyoumax/QAsystem-NLPT-WS2023 (github.com)](https://github.com/lingyoumax/QAsystem-NLPT-WS2023/tree/main/Question_type)) to select 30 representative seed questions from each category, resulting in 164 seed questions. It is worth noting that because there are relatively few variations of Confirmation Questions, the number of seed questions we selected in this category is less than 30.

2. Template sampling structure and problem screeningIn this step, we decide the composition of the template based on the presence or absence of newly generated questions. If there are no newly generated questions, we select 8 from the seed questions as templates; if there are, we select 6 from the seed questions and 2 from the newly generated questions. Then, one or several semantically segmented texts are selected as content to generate fine-grained and coarse-grained problems. We set up to generate two questions for each text, so that an abstract can generate approximately six to eight questions. Newly generated questions need to be further screened, first by calculating the Rouge value[^17] with the existing question set to exclude similar questions (the threshold is set to 0.7), and secondly by format filtering to ensure that the questions are content-based and formatted correctly.

3. Generation of answersThe answer generation process is relatively straightforward. We choose the answer generated based on GPT-3.5 as the standard answer to ensure the accuracy and reliability of the answer.Through the above steps, we can effectively improve the quality and diversity of the data set and provide a solid foundation for the training and evaluation of question answering systems.

##### **MoDS** 

After constructing the data set generated by the self instruct method, we faced the challenge of huge data volume and poor quality of some data. In particular, in the initial dataset, we observed that many instances had questions that were not relevant to the content, affecting the overall quality of the dataset. Considering that data quality is more important than data quantity for SFT, we adopt MoDS as an effective solution.

The MoDS method is not only simple and efficient, but also significantly improves data quality, ensuring that the data used in the training process is more accurate and relevant. In this way, we can filter out high-quality examples from huge data sets, optimize training results, and reduce training time while ensuring data quality.

In our MoDS approach, we take three key steps to optimize the training dataset required for SFT, aiming to balance data quality and diversity, as well as enhance dataset coverage. Specific steps are as follows:

1. Selection of high-quality answersFirst, we use [reward-model-deberta-v3-large-v2]([OpenAssistant/reward-model-deberta-v3-large-v2 · Hugging Face](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2)) to score the answers in the data set and filter out the answers with a score higher than 0 as high-quality data. This step ensures that the answers contained in the selected data set are of high quality and provides a solid foundation for model training.

2. Diverse data selection for seed instructionsAfter obtaining high-quality answers, we obtain the embedding vectors of sentences through the bert-base-uncased model, and use a greedy search method to select the top 2000 instances with the most diverse vector representations to form a diverse data set. This step aims to increase the coverage and diversity of the data set and improve the model's generalization ability to different types of data.

3. Selection of enhanced data setsFinally, we fine-tune the pre-trained large language model (LLM) through the above diverse dataset to improve its performance. Subsequently, inference is performed on those data that exist in the high-quality dataset but not in the diverse dataset, use reward-model-deberta-v3-large-v2 to score these inference results, and select instances with lower scores as enhanced datasets . This step further optimizes the training data set by supplementing it with low-quality or challenging examples, ensuring that the model still performs well in the face of complex or low-quality input.After the above steps, we successfully constructed an SFT training data set containing 3818 data points. This data set not only has high quality and good diversity, but also contains targeted enhanced data, laying the foundation for efficient and accurate few-sample training. Solid foundation.

#### **Introduction to basic models**

##### **Base model** 

After comprehensively considering our training resources, data set size, training time and other limiting factors, we determined that the maximum size of the training model is 7B parameters. For our closed QA task, which involves answering a question after retrieving relevant text, this process places specific demands on the model's ability to handle context length. Based on these considerations, we chose Qwen-7B-Chat as the basic model, whose context length can reach 8192[w5]. Not only did this model accommodate our hardware and data scale constraints, but it was designed and optimized with the need for longer context processing in mind, making it ideal for performing our closed-loop question answering tasks.

Qwen adopts an improved version of Transformer architecture. Specifically, the training method of the recently open source large language model LLaMA is adopted.

##### **QLoRA**

However, in fact, if the 7B model is used directly, the video memory occupied is also very large. So we used QLoRA to read and fine-tune the model.
QLoRA technology proposes an efficient strategy for fine-tuning pre-trained language models, aiming to optimize the storage and computing efficiency of the model. Specifically, this strategy first uses 4-bit quantization technology to compress the model to reduce storage requirements and speed up calculations. In this process, we use a new data type - 4-bit NormalFloat, and combined it with Double Quantization and Paged Optimizers technology to further optimize the storage and computing efficiency of the model[w4].

In the fine-tuning phase, the QLoRA strategy freezes the parameters of the pre-trained model and only adds a small number of trainable low-rank adapters to the model, especially in all fully connected layers, to provide additional training capabilities. In this way, only the weights of the LoRA layer need to be updated, greatly reducing the amount of parameter updates during the training process. The addition of these low-rank adapters not only makes up for possible performance degradation due to accuracy loss, but also maintains the flexibility and scalability of the model.

1. 4-bit NormalFloat, double quantization and paging optimizerThe 4-bit NormalFloat is designed to reduce the storage requirements of model parameters while maintaining calculation accuracy. By estimating and applying the $2^{k+1}$ quantiles of the $N(0, 1)$ distribution, this method can effectively quantize the weight parameters into a $k$-bit representation range. During the quantization process, the weight parameters are first normalized to the range of [-1, 1], and then further quantized through an absolute maximum recalibration technique. The mapping formula used in this process is as follows:

$$
q_i=\frac{1}{2}(Q_x(\frac{i}{2^k+1})+Q_x(\frac{i+1}{2^k+1}))
$$

This strategy not only reduces the storage requirements of the model, but also enables the model to run at lower  	computational precisions (such as BFloat16) without significantly affecting performance.

2. Double quantization technology further reduces the storage space of quantized parameters, and optimizes storage efficiency by quantizing the quantized constants twice.

3. The paging optimizer takes advantage of NVIDIA's unified memory feature, allowing automatic data transfer between the CPU and GPU when the GPU memory is insufficient, ensuring the continuity and stability of model training and avoiding out-of-memory overflow (OOM). Training interrupted.

4. QLoRA reduces the amount of parameter updates by inserting low-rank adapters in the fully connected layers of the model, especially in the fully connected layers of query and value. This strategy not only optimizes the training efficiency of the model, but also improves the performance of the model after quantification. Through carefully designed low-rank adapters, QLoRA can effectively improve the training and inference performance of the model while maintaining model size and computational efficiency.

In summary, QLoRA provides an effective strategy for fine-tuning pre-trained language models. Through technologies such as 4-bit quantization, dual quantization, paging optimizer, and LoRA's low-rank decomposition, it achieves a balance between model performance and computing and storage efficiency. the optimal balance between.

#### **First fine-tuning**

By using the above technology and the obtained data set, we can perform the initial fine-tuning process and obtain the first version of the supervised fine-tuned model SFT_v0 (Supervised Fine-Tuning version 0) and the first version of the reward model rm_v0 (Reward Model Fine-Tuning version 0). Tune version 0). This section will outline the fine-tuning strategy.

##### Supervised Fine-Tuning

At this stage, we adjust the dataset format by replacing the "content" field with "instruction", the "question" field with "input", and the "answer" field with "output" to conform to the format of the standard dataset. The specific training process includes the following steps:

Data preprocessing: First, read the "instruction", "input" and "output" fields, splice these three parts together, and unify the sequence length through padding or truncation operations. At the same time, a mask is generated to identify the output part when calculating the subsequent loss, that is, the mask value of the output part is set to 1 and the remaining parts are 0.

Model training: Convert the preprocessed data into the corresponding embedding representation (embedding) through the tokenizer, and then calculate the output through the model. By comparing the model output with the "output" in the dataset, a cross-entropy loss with a mask value of 1 is calculated and backpropagated to update the model parameters.

#### Reward Model Fine-Tuning

During the fine-tuning process of the reward model, we use the SFT_v0 obtained through supervised fine-tuning as the basis, remove its language model head (lm_head) and replace it with a fully connected layer (full linear layer) to form the reward model. This process mainly focuses on the following two aspects:

##### Dataset construction

After using the self instruct method to generate the question data set, SFT_v0 was used to randomly generate 4 answers, combined with the 2 answers generated by GPT-3.5, for a total of 6 answers.

By using GPT-3.5 to score each answer, selecting the answer with the highest score as "chosen" and the answer with the lowest score as "rejected", a data set containing 2,000 pieces of data was constructed.

##### Training process

Data preprocessing: First, adjust the "chosen" and "rejected" sequences to the same length through padding or truncation operations. Then, connect the "chosen" and "rejected" sequences with prompts respectively to form a complete data record.

##### Loss calculation

Different from the previous output of calculating the probability of each position, the output of this model is the value of the sequence. The loss is calculated using the following formula, the value in the middle from the first inconsistent position of the two sequences to the padding position as the end will be selected as the score:
$$
loss=-mean(logsigmoid(score_1-score_2))
$$

##### Inference process

During inference, we select the score of the last position of the non-filler characters of each sequence as the evaluation criterion. This approach is based on the principle that the reward at the end of the sequence reflects the effect of the entire sequence, that is, the reward at the end of the sequence often accumulates the results of the entire sequence.

The fine-tuning strategy outlined above aims to improve the performance and adaptability of the model on specific tasks by accurately adjusting model parameters and optimizing the reward model.

#### **pipeline training**

After the model constructed previously, we can start to update the model semi-automatically.Here we use two methods, one is the rejection sampling method, and the other is the DPO reinforcement learning method used after the rejection sampling iteration.

##### **[reject sampling](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)**

Here we use a similar training method to llama2. After using the self instruct method to generate the question and answer data set, we used the SFT_v0 model to generate five answers. To ensure quality, the first answer is generated by using a stable parameter configuration, while subsequent answers are generated by using a wide range of random parameters to introduce diversity. Next, we score these responses using a preliminary fine-tuned reward model (rm_v0). The answer with the highest score is selected as the standard answer to further fine-tune SFT_v0. After one or two rounds of iterative fine-tuning, we obtained SFT_v1 - which refers to an improved version of the hypothetical model that has undergone one round of fine-tuning. Finally, we use the high-quality responses (gold responses) generated by SFT_v1 to further fine-tune the reward model rm_v0.

This process not only optimizes the quality of answers, but also continuously improves the performance of the model through iterative fine-tuning. By introducing random parameters in the iterations, the model can explore a wider answer space and guide the learning process by rewarding the model with scores that ultimately tend to generate optimal answers. This approach promotes specialization of the model on a specific task, making it more precise and reliable in generating answers.

However, due to time reasons, this step was not fully iterated and only iterated to SFT_v2 and rm_v0.

##### **DPO**

After constructing the training data set using the rejection sampling method, we adopted the DPO method to further fine-tune the learning process to make the answers generated by the model closer to high-quality human answers. The main reason for choosing DPO instead of PPO is that DPO is more economical in resource consumption. Its core advantage is to increase the logarithmic probability difference between the preferred answer (chosen) and the inferior answer (rejected). The content of this part mainly focuses on two core parts: the construction of the data set and the definition of the loss function.

###### Construction of data set

The data set construction of this study is similar to rejection sampling, but there are slight differences in the specific implementation. Specifically, we compare the answers generated by the v_k version of the model with the v_{k-1} version for the same question, and use the reward model to score these answers to select the highest-scoring and lowest-scoring answer pairs, forming a chosen-rejected pair that serves as the basis for a training data set.

###### Composition of loss function

The loss function is defined as follows:
$$
L_{DPO}(\pi_{\theta};\pi_{ref})=-\mathbb{E}{(x,y_w,y_l)~D}[log \sigma (\beta log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\beta log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})]
$$
Among them, y_w, y_l respectively represent higher-quality answers and lower-quality answers in a certain preference data, 

$\pi_\theta(y_w|x)$ 

represents the preferred answer generated by the current strategy model when the input x is given. The cumulative probability of , and $\pi_{ref}(y_l|x)})$ represents the cumulative probability of the reference model (i.e. the original model) generating a poor answer when the input x is givenIn essence, our approach involves a reference model and a policy model, where the weights of the reference model are not updated. By transforming the loss function, we get the following form:This loss function update strategy aims to increase the probability of preferred answers while reducing the probability of inferior answers, thereby optimizing model performance and making the answers it generates closer to high-quality human answers.

## **Experimental Setup**

### **Data preparation**

#### Data Acquisition

For the "Data Acquisition" phase of our project, we adhered closely to the guidelines provided in the project documentation regarding data selection. Our team elected to focus on PubMed as our primary source of data. Specifically, we targeted abstracts from articles published between 2013 and 2023 that include the term "intelligence" within the abstract text. This decision aligns with our project's domain of interest and sets a clear foundation for our research objectives.

To efficiently gather the required data, we utilized [Entrez Direct](https://www.ncbi.nlm.nih.gov/books/NBK179288/), a tool recommended by PubMed. Entrez Direct offers a command-line interface for accessing NCBI's comprehensive databases, allowing us to precisely retrieve the needed abstracts. This method ensures a streamlined and effective data collection process, pivotal for the success of our subsequent analyses. 

#### Abstracts Dataset

**Purpose**: Serves as the foundational corpus for the entire project.

**Description**: Abstracts from articles published between 2013 and 2023 that include the term "intelligence" within the abstract text.

**Source**: Retrieved from PubMed.- 

**Collection Methods**: Utilized the Entrez Direct tool to efficiently query and download relevant abstracts based on specific search criteria, including date range and keyword occurrences. - 

**Preprocessing Steps** 

**Removal of Rows with Missing Values**: Any rows in the CSV files containing missing or incomplete information were identified and removed to ensure the integrity and consistency of the dataset. 

**Exclusion of Retracted Articles**: Articles that have been retracted were systematically identified and excluded from the dataset. 

**Text Splitting**: Employed the `NLTKTextSplitter` from the `langchain.text_splitter` package to segment the abstracts into smaller, manageable chunks. This step is crucial for enhancing the processing efficiency and aligning with the input requirements of subsequent models.



#### Initial Question-Answer Pairs Dataset

**Purpose**: Serves as the foundational corpus for the entire project.

**Description**: Generate some question and answer pairs based on the article information as the initial training dataset

**Source**: Generated from abstracts dataset

**Collection Methods**: Use the GPT-4 api and send it a prompt with instructions and references



#### Fine-Tuning Dataset for Text Retrieval Model in Semantic Search

**Purpose**: Used to fine-tune the embedding model for enhanced performance in text retrieval tasks within our project's specific domain.- 

**Description**: This dataset is crafted by further processing the Initial Question-Answer Pairs Dataset, in alignment with the structured methodologies outlined in the [FlagEmbedding fine-tuning examples](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune).

**Source**: Derived from the Initial Question-Answer Pairs Dataset.

**Collection Methods**: 

**Methodological Alignment**: Followed the structured approach and best practices as recommended in the FlagEmbedding fine-tuning examples for processing and preparing the dataset. 

**Extraction and Transformation**: Questions and answers were extracted from the Initial Question-Answer Pairs Dataset and transformed into a format suitable for embedding model training, including pairing questions with relevant text passages. - 

**Quality Filtering**: Employed a rigorous quality filtering process to ensure the inclusion of only high-quality, relevant question-answer pairs, maintaining the integrity of the model training process. 

**Hard Negatives Mining**: Implemented the hard negatives mining technique as suggested in the FlagEmbedding examples. This method enhances the quality of sentence embeddings by identifying and incorporating challenging negative examples that are close to positive samples in the embedding space but are not relevant. This approach significantly improves the model's discrimination capabilities.-

**Preprocessing Steps**: 

 **Manual Filtering**: Conducted a general manual filtering of the Initial Question-Answer Pairs Dataset to remove those that did not meet our criteria for high quality and relevance. This process involved a broad review rather than an exhaustive examination of each pair, focusing on identifying and excluding obviously unsuitable content. 

**Dataset Merging**: The Initial Question-Answer Pairs Dataset, which comprised only three columns (Question, Answer, and PMID), were merged with the unsplit Abstracts Dataset. This step was necessary because the text associated with each PMID in the question-answer pairs was in its original, unsplit form. By merging these datasets using PubMed IDs, we were able to link each question-answer pair with the full, corresponding abstract text from the Abstracts Dataset, providing a complete context for further processing and analysis. 

 **Text Splitting and Embedding**:   Employed `NLTKTextSplitter` from `langchain.text_splitter` to segment the merged abstract texts into smaller chunks. Utilized [voyageai](https://docs.voyageai.com/embeddings/) to generate embedding vectors for each chunk and the questions.   Calculated cosine similarity between question vectors and chunk vectors to identify the top 3 most similar chunks for each question. 

 **Training Data Preparation**:   Formatted the fine-tuning dataset as a JSON file, where each line is a dictionary with the structure: `{"query": str, "pos": List[str], "neg":List[str]}`.   Here, `query` represents the question, `pos` includes the top 1 chunk (most similar to the question), and `neg` is a randomly chosen chunk from the remaining ones. 

**Hard Negatives Mining**: Applied the hard negatives mining technique as outlined in the FlagEmbedding fine-tuning examples to further enhance the dataset's quality by incorporating challenging negative examples that closely resemble positive samples in the embedding space.

### Seed question dataset

**purpose**:Used to generate the seed data set for all subsequent fine-tuning training.

**Description**: A data set from our own public data set that has been filtered to be more representative of the questioning methods.

**Source**: databricks-dolly and SQuAD2

**Data Format**: [dict],which have two columns:[Question,type]

**Collection Methods**:

i.Collect all question parts from databricks-dolly and SQuAD2.

ii.Convert data to embedding via bert-base-uncased.

iii.Select random data as the starting point, and successively select the points farthest from the selected point to add to the selected points.

iv.To form a sub-dataset, use a question type model to obtain a certain number of data sets for each category as a seed data set.

### Self instruct dataset

**purpose**:A data set of question answer pairs generated based on content awaiting further processing.

**Description**: QA pairs based on PubMed dataset generated using self instruct method.

**Source**: Dataset generated from seed question**Data Format**: [dict],which have three columns:[Question,Content,Answer]

**Collection Methods**:The generation method can be seen in the previous method self instruct

### SFT_v0 train dataset

**purpose**:Use the training data set for the subsequent SFT process.

**Description**: The data set generated by self instruct is further filtered through the MoDS method to obtain the final training data set.

**Source**: Self instruct dataset.

**Data Format**: [dict],which have three columns:[Question,Content,Answer]

**Collection Methods**:The generation method can be seen in the previous method MoDS.

### Reward model train dataset

**purpose**:Data set suitable for reward model training

**Description**: None

**Source**: GPT3.5 and SFT_v0.

**Data Format**: [dict],which have three columns:[prompt,chosen,rejected]

**Collection Methods**:i.Use self instruct to generate a question answer pair, where the answer comes from the answer generated by GPT3.5, and then use SFT_v0 to generate a response to the question.ii.Send two answers to GPT3.5 to get chosen and rejected

### Reject sampling dataset

**urpose**:Data set used to further fine-tune SFT_vk

**Description**: None

**Source**: GPT3.5 and SFT_vk.

**Data Format**: [dict],which have three columns:[instruction,input,output]

**Collection Methods**:i.First generate the question dataset through self instruct.ii.Use the SFT pre-trained model to generate random parameters, generate 5 answers, and use the reward model for scoring.For each question, select the highest scoring response to form answer question pairs.

### DPO train dataset

**purpose**:Data set used to further fine-tune SFT_vk

**Description**: None

**Source**: GPT3.5, SFT_vk, SFT_vk-1 or base model.

**Data Format**: [dict],which have three columns:[prompt, chosen, rejected]

**Collection Methods**:i.First generate the question dataset through self instruct.ii.Use the latest model to generate 2 responses, use the model of the previous version to generate 2 responses, and use the reward model for scoring.Select the highest score and the lowest score to form chosen, rejected pairs   

### Eveluation dataset

**purpose**:Data set used to evaluate models

**Description**: None

**Source**: GPT-3.5, GPT-4

**Data Format**: [dict],which have three columns:[instruction, input, output]

**Collection Methods**:i.First generate the reference-question pairs dataset through self instructii.Use the GPT-4 to generate the answer for each reference-question pair and combine them into a dataset

All data relevant to this project, including datasets for training, evaluation, and any supplementary materials, can be obtained [here](https://drive.google.com/drive/folders/1aPfA09s0cbpxDb_nI3MZzfm5bUZE_0Ie?usp=sharing).

### Dataset analysis metrics

We reflect the diversity of information in the dataset by analyzing the length of the text and the word cloud of the text.

#### SFT_v0 train dataset

#### Reject sampling dataset

#### Evaluation dataset

From the histogram above, it can be seen that the length distribution of the question and answer in our designed dataset is consistent. The length of references varies slightly, which may be due to the different lengths of articles involved in different issues. From the word cloud, it can be seen that these datasets target almost identical keywords. From the analysis of these two types of graphs, our dataset maintains consistency and diversity in information.

## **Evaluation Method:**

#### Text Retrieval Evaluation

Top-1 Accuracy: Due to our question generation process solely utilizing a single abstract section, the number of relevant sections remains unknown. Consequently, evaluation metrics like recall become inapplicable. We opt for a straightforward approach: the retrieved answer's agreement with the abstract section used for question generation determines its correctness. Subsequently, the top-1 accuracy metric is calculated as the ratio of correctly answered questions to the total number of tested questions.

### Answer generation Evaluation

#### Answer Correctness Score:

In order to measure the correctness of the model's answers, we will use the answers generated by GPT-4 under the same input conditions as reference answers. When selecting a model to evaluate the correctness of answers, we first consider several aspects, enhanced semantic understanding, context sensitivity and flexibility in language. They can enable our evaluation to better focus on the understanding of semantic and contextual relationships between words, and ideally not rely on strict matching between words. Due to BERTScore's [^18] better performance in these areas, we ultimately chose it as our evaluation model. The process of this evaluation method is shown in the following figure.

#### Safety Score:

"Toxic responses" is a common phenomenon where natural language processing (NLP) models, such as chatbots or text generation models, produce harmful, inappropriate, or offensive content during text generation. These responses can include discriminatory speech, hate speech, misleading information, insulting language, or any other form of negative content. Toxic responses may arise due to biases or undesirable information present in the model's training data, as machine learning models tend to learn and replicate the patterns and preferences found within their training datasets. So ensuring the safety of the model to users is very important. 
We used the [reward-model-deberta-v3-large-v2](https://huggingface.co/manueldeprada/FactCC)) model from Hugging Face to aid in assessing the safety of our model. It scores pairs of questions fed into the model and the answers produced by the model, with question-answer pairs exhibiting fewer toxic responses receiving higher scores. The average score of all question-answer pairs obtained after testing the model with a dataset is considered the safety score of the model. The process of this evaluation method is shown in the following figure.

#### Factual Consistency Score

In the field of natural language processing (NLP), especially in tasks such as text generation, summarization, or question-answering systems, "factual consistency" is paramount. It measures the extent to which the generated text or provided answers correspond to the facts presented in the source material or real-world knowledge. Assessing a model's factual consistency is crucial for ensuring the credibility and integrity of information, and it also enhances users' trust in the model. In our model, an important task is to verify whether the answers generated by the model can find concrete evidence in the reference material. If this task were to be done manually, it would not only be time-consuming but also challenging to quantify this metric objectively due to human evaluators' biases and subjectivity. Therefore, we have employed the FactCC model[^20] from the Hugging Face platform. This model is capable of scoring the reference content input to the model and the answers it generates, with higher scores indicating greater consistency between the two. The average score of all reference-answer pairs constitutes the factual consistency score of the model. The process of this evaluation method is shown in the following figure.

#### Spelling Error Rate

Ensuring the spelling accuracy of answers generated by question-answering models is a critical aspect of evaluating model performance, as it helps prevent user misunderstandings. The better a model performs in terms of spelling accuracy, the more likely it is to gain higher trust from users. To accomplish this task, we have employed the TextBlob library in Python. This library is capable of identifying spelling mistakes in the answers and suggesting corrections. By comparing the answers before and after correction, we are able to calculate the spelling error rate for answers generated by each model, thereby assessing their performance. The process of this evaluation method is shown in the following figure.

## **Experimental Details:**

### **Text Retrieval**

#### Semantic Search

##### Data Indexing

 **Data Source**: The data was read from a CSV file named `splitted_pubmed_data_NLTK`, containing segmented abstracts alongside metadata such as PMIDs, publication dates, titles, and authors. 

**Mapping and Schema Definition**: For our vector database within Pinecone, we define a schema focusing on both embeddings and their associated metadata for efficient storage and retrieval. 

This schema includes: 

**Unique Identifier (`id`)**: Combines PubMed Identifier (`pmid`) and a text chunk identifier (`text_chunk_id`) for uniqueness. - **Vector Embeddings (`values`)**: Stores the embeddings generated from text chunks. 

Metadata: Encompasses several** key pieces of information:  - **`pmid`**: Unique identifier for scientific publications.  - **`title`**: Article title.  - **`publishedDate`**: Publication date of the article.  - **`authors`**: List of article authors.  - **`text_chunk_id`**: Identifier for the specific text chunk within an article.  - **`arxiv_text`**: The actual text chunk content.

**Indexing Process**: Each row from the dataframe was indexed into Pinecone, with the text chunks encoded into embeddings using the `FlagModel.encode` method before indexing.

##### Fine-Tuning Process

The model was fine-tuned using a script that leverages PyTorch's distributed training capabilities. Key parameters included: - **Learning Rate**: Set to `1e-5`, a commonly used rate that balances convergence speed and stability. 

**Batch Size**: A per-device train batch size of 4 was used, considering the balance between computational resource constraints and training efficiency. 

**Epochs**: Training was conducted for 1 epoch to apply a moderate level of fine-tuning to the pre-trained embeddings. 

**Max Token Lengths**: The maximum lengths for queries and passages were set to 120 and 408, respectively, to accommodate the typical length of inputs while respecting the model's capacity. 

**Temperature**: Set to `0.02`, affecting the sharpness of the softmax distribution used in contrastive learning, with a lower temperature leading to a sharper distribution.

##### Combined PRF Search

For the initial search, we only use the top1 match abstract from Hierarchical Search and semantic search as the input of the second stage. The results of the search are not the whole abstract, but the chunks of related abstract, which is [NLTK dataset]().

### **Answer Generation:** 

Self instruct process:Key parameters:

**Engine**: GPT-3.5-turbo-instruct or qwen-max-1201

**seed_question_path**: Use the path of the seed question.

**Num_prompt_instructions**: default 8, The number of questions used as reference

**num_instructions_to_generate**： We have tried to generate up to 6000 pieces of data.If use qwen-max-1201, it is free, but it takes a long time because batch inference is not developed.If use GPT3.5, it will cost 0.8 euros for every 2,000 questions and 0.8 euros for 2,000 answers, but the speed will be much faster.

Other information:

**Environment**: 3060 laptop, 6GB

**time**: 2000 questions, qwen-max-1201~4h, GPT3.5~30 min

**MoDS:**Key parameters:

**Model_name**:The model used to convert embedding.

**Reward_model_name**:Models used to differentiate between higher quality data sets threshold: default 0，Select data above the threshold.

**Top_k**：default 2000, select k diverse data

**finetune_model**:fine-tuned model

**Argument_threshold**: default 0, select data selection parameters that do not diversify the data but perform fine-tuning on high-quality data with a score below the threshold.

**SFT_train:**Key parameters: **model_name** the path of base model or name in hugging face.

**Adapter**: default None, If it is None, it indicates that an adapter is initialized and the path is specified, and the lora file for the path will be loaded.

**Dataset_path**： train dataset path, the file should be json format.Train_args: the parameters of trainer setting. Should be TrainingArguments format. In this part, we use learning_rate=1e-4,num_train_epochs=2, batch size=12,other parameters such as the optimizer use the default parameters.Save_dir: the finetune lora file save path.

Other information:

**Environment**: 4090, 24GB

**time**: about 1h
RM_train:Key parameters:

**model_name:** the path of the base model or name in the hugging face.

**Adapter**: default None, If it is None, it indicates that an adapter is initialized and the path is specified, and the lora file for the path will be loaded.

**Dataset_path**： train dataset path, the file should be json format.Train_args: the parameters of trainer setting. Should be TrainingArguments format. In this part, we use learning_rate=1e-4,num_train_epochs=1, batch size=8(because each inference needs to infer chosen and rejected together, the batch needs to be smaller)other parameters such as the optimizer use the default parameters.Save_dir: the finetune lora file save path.

Other information:

**Environment**: 4090, 24GB

**time**: about 1h
Reject_sampleKey parameters:finetune_config:All parameters applicable to pipeline.Finetune_steps: default [1,2,3,4]，1-self instruct, 2-generate response, 3-score and choose highest score, 4-SFT_traingenerate response parameters: The first temperature and top_p are set to 0.7 and 0.2 respectively to generate a more stable answer, and subsequent other answers are set to a uniformly distributed random number from 0 to 1,response num=5.SFT_train parameters: learning_rate=1e-4, per_device_train_batch_size=24, num_train_epochs=2

Other information:

**Environment**: 4090, 24GB

**time**: about 7h for 2000 train samples
DPOKey parameters:finetune_config:All parameters applicable to pipeline.Finetune_steps: default [1,2,3,4]，1-self instruct, 2-generate response, 3-score and choose pairs answers, 4-DPO_traingenerate response parameters: The first temperature and top_p are set to 0.7 and 0.2 respectively to generate a more stable answer, and subsequent other answers are set to a uniformly distributed random number from 0 to 1, response_num=2.DPO_train parameters: learning_rate=1e-4, per_device_train_batch_size=12, num_train_epochs=2

Other information:

**Environment**: A100, 40GB

**time**: about 9h for 2000 train samples

## **Results and** Analysis

### **Text Retrieval**

We test top1-Accuracy evaluation metrics on the original BM25 algorithm, Hierarchical BM25 algorithm, fine-tuned bge model, and Combined PRF Search.

|                      | Top1-Accuracy |
| -------------------- | ------------- |
| BM25                 | 0.71          |
| Hierarchical BM25    | 0.74          |
| bge model            | 0.70          |
| Fine tuned bge model | 0.75          |
| Combined PRF Search  | 0.79          |

Our retrieval system underwent evaluation with several models, employing the top 1-Accuracy metric. This assessment encompassed the original BM25 algorithm, its Hierarchical variant, a fine-tuned Background Expansion (bge) model, and our distinctive Pseudo-Relevance Feedback (PRF) Search.

The BM25 algorithm, serving as the foundational benchmark, achieved a top 1-Accuracy of 0.71. This score establishes a robust baseline for information retrieval efficacy.

In an extension of BM25, the Hierarchical BM25 algorithm, which presumably integrates document hierarchy into its ranking mechanism, demonstrated a modest improvement with a top 1-Accuracy of 0.74.

Before fine-tuning, the bge model achieved a top 1-Accuracy of 0.70. This result positions it closely with the foundational BM25 algorithm, suggesting potential for significant improvement upon optimization.

The fine-tuned bge model, likely augmented with techniques for expanding background knowledge, attained a top 1-Accuracy of 0.75. This indicates its superior ability to rank relevant documents more accurately.

Our PRF Search model, innovative in its approach by intersecting results from dual sub-PRF queries, recorded the highest top 1-Accuracy at 0.79. This superior performance underscores the model's effectiveness in identifying the most relevant document at the top rank. The intersecting strategy of our PRF queries presupposes that the confluence of relevant document sets from separate sub-queries more precisely targets the most pertinent information.

The exceptional top 1-Accuracy of our PRF Search model underscores its advanced proficiency in sourcing the most relevant document for user queries. This capability is especially valuable in scenarios where promptly delivering the most pertinent response is imperative.

### **Answer Generation** 

Through the use of evaluation datasets in question and answer tasks, we compared and analyzed several common models as well as our own customized models. This table includes well-known models such as BERT, GPT-3.5, and a model named Yi, as well as our NaiveModel (used as an unadjusted baseline) and other fine-tuning models. The evaluation results are shown in the table below:

![image-2024030474305731 PM](/Users/liuziwei/Library/Application Support/typora-user-images/image-2024030474305731 PM.png)

The GPT-3.5 model stands out with a perfect score in precision, recall, and F1, indicative of its exceptional ability to provide accurate and complete answers. Our fine-tuned model, SFTV0 Model, is second only to it in terms of the correctness of the answers.

Regarding the safety of generating answers, except for BERT, other models perform well in one aspect. Even if the safety score of individual models is less than 0, after careful examination of the results, it was found that this is acceptable.

BERT's performance in factual consistency is quite high. However, upon inspection, I've realized this is due to its strategy of extracting words directly from the reference material to form very short answers. This approach, while boosting factual consistency, may not always provide comprehensive or nuanced responses, which is something to consider when evaluating its practical application.

Except for BERT, all other models perform almost the same in terms of word spelling. It is worth noting that our fine-tuning model, SFTV0 Model, have slightly regressed compared to the NaiveModel in this regard.

In conclusion, the fine-tuning of my models has resulted in a notable improvement in overall performance, particularly in the balance between precision, recall, and safety. The high consistency between the answers and references generated by SFTV0Model indicates a deeper understanding of the text. Furthermore, it stands out as a strong contender in question-answering tasks, with a performance that is comparable to GPT-3.5 in terms of accuracy, which means that it has broad application areas. And DPO Model performs well in safety, making it a promising model for applications that require a high degree of safety in automated responses.

## **Conclusion and Future Work**

### Conclusion

Our project effectively utilized advanced fine-tuning techniques and dataset optimization strategies, demonstrating significant performance improvements in our pre-trained model. Key strategies included prioritizing keywords through a BM25 variant, hierarchical searching, and the combined use of pseudo-relevance feedback to align user queries with document topics. The fine-tuning of the bge model underscored the importance of model optimization, with extensive fine-tuning operations validating our approach. The critical role of high-quality datasets in the training process emerged as a pivotal finding, highlighting the necessity for meticulous dataset optimization.

### Future Work

Looking forward, we plan to enhance our system by:

Advanced Keyword Extraction: Implementing named entity recognition and domain-specific tools for more accurate keyword identification.

Dynamic BM25 Weighting: Adapting weighting strategies based on query specifics to improve search precision.

Deeper Semantic Integration: Employing sentence embedding or topic modeling for a fuller understanding of document content.

Efficient Evaluation Methods: Adopting more precise evaluation techniques to streamline the fine-tuning process.

Data Quality Improvement: Seeking higher quality data sources or refining the data generated by GPT-3.5 to ensure a robust foundation for model training.

Model Innovation: Continuously updating and innovating our model with new architectures, training strategies, and technologies to stay at the forefront of AI research.

These future directions aim to address current limitations and propel our research forward, enhancing the effectiveness and reliability of our AI models.

## **Reference**

[^1]:Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems 33 (NeurIPS 2020).
[^2]:Robertson, S. E., Walker, S., Jones, S., Hancock-Beaulieu, M. M., & Gatford, M. (1994). Okapi at TREC-3. In Proceedings of the 3rd Text Retrieval Conference (TREC-3).
[^3]: Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., & Hajishirzi, H. (2022). Self-instruct: Aligning language model with self generated instructions. arXiv preprint arXiv:2212.10560
[^4]:Du, Q., Zong, C., & Zhang, J. (2023). Mods: Model-oriented data selection for instruction tuning. arXiv preprint arXiv:2311.15653.
[^5]: Can foundation models label data like humans? [huggingface.co](https://huggingface.co/blog/open-llm-leaderboard-rlhf)
[^6]:Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2024). Qlora: Efficient finetuning of quantized llms. Advances in Neural Information Processing Systems, 36.
[^7]:Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., ... & Zhu, T. (2023). Qwen technical report. arXiv preprint arXiv:2309.16609.
[^8]: Li, Z., Li, X., Liu, Y., Xie, H., Li, J., Wang, F. L., ... & Zhong, X. (2023). Label supervised llama finetuning. arXiv preprint arXiv:2310.01208.
[^9]:Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., & Finn, C. (2024). Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36.
[^10]:Lavrenko, V., & Croft, W. B. (2001). Relevance-based language models. In Proceedings of the 24th annual international ACM SIGIR conference on Research and development in information retrieval.
[^11]:Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of NAACL-HLT 2019.
[^12]:Lewis, P., Oguz, B., Rinott, R., Riedel, S., & Schwenk, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Proceedings of NeurIPS 2020.
[^13]: Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
[^14]:Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9.
[^15]:Carpineto, C., & Romano, G. (2012). A survey of automatic query expansion in information retrieval. ACM Computing Surveys (CSUR), 44(1), 1-50.
[^16]:Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
[^17]:Lin, C. Y., & Och, F. J. (2004, June). Looking for a few good metrics: ROUGE and its evaluation. In Ntcir workshop.
[^18]:Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2019). Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675.
[^19]:Kryściński, W., McCann, B., Xiong, C., & Socher, R. (2019). Evaluating the factual consistency of abstractive text summarization. arXiv preprint arXiv:1910.12840.



























