#!/bin/bash

# 定义要搜索的关键词和年份范围
QUERY="intelligence[Title/Abstract] AND (\"2013/01/01\"[Date - Publication] : \"2023/12/31\"[Date - Publication])"

# 使用 esearch 查找符合条件的论文，并提取它们的 PubMed ID
esearch -db pubmed -query "$QUERY" |
efilter -pub abstract |
efetch -format uid > pubmed_ids_intelligence_2013_2023Id.txt



#esearch -db pubmed -query "intelligence[tiab] AND 2013:2023[dp] AND hasabstract" | efetch -format uid > pubmed_ids_intelligence_2013_2023Id.csv