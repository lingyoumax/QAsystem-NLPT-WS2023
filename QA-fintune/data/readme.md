# 1 Description of the data set
Our dataset is abstracts from papers downloaded from PubMed. The questions and answers come from GPT4 and parts collected by myself.
# 2 Dataset format
- I.raw data:csv format, including [question, answer, abstract]
- II.The first conversion format:json format,by converting the csv format in the previous step to json format, including [instruction:xxxx(question part),input:xxxx(abstract part),answer:xxx(answer part)]
- II.train data:Finally, json is converted into a format that conforms to hugging face dataset reading.
