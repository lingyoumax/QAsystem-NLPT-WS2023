## Evaluate with LLM algorithm

Question - Reference Paragraph -> Answer 

### Exploration Indicators:

##### Stability and Consistency: 

Whether the output expresses the same meaning each time. 
Measure the cosine similarity of the embedding vectors of two output answer sentences.

##### Relevance:

Whether the answer can fully and correctly answer the question. 
Measure the cosine similarity of the embedding vectors of the question and answer sentences + textual entailment.

##### Credibility:

Whether the answer can be derived entirely from the reference paragraph of the question. 
Measure the cosine similarity of the embedding vectors of the answer and reference paragraph sentences + textual entailment.

##### Naturalness:

The coherence of the answer sentence.

##### Conciseness:

Whether it contains content other than the question. 
Information density in the answer + compare the total content and the original answer's cosine similarity and length after text summarization technology.

##### Response length
We can use the current GPT4 responses as a response template and then compare the response lengths of the different models
