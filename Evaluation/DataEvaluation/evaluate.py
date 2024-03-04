import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from getData import getData

data=getData('dataset/eveluateData.json')
data=data[:100]
references=[item['instruction'] for item in data]
questions=[item['input'] for item in data]
answers=[item['output'] for item in data]

len_references=[len(reference) for reference in references]
len_questions=[len(question) for question in questions]
len_answers=[len(answer) for answer in answers]


fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # figsize控制总图形的大小

# 分别绘制每个子图
axs[0].hist(len_references, bins='auto')
axs[0].set_title('Reference Length Distribution')  # 设置第一个子图的标题
axs[0].set_xlabel('Length')  # 设置X轴标签
axs[0].set_ylabel('Frequency')  # 设置Y轴标签

axs[1].hist(len_questions, bins='auto')
axs[1].set_title('Question Length Distribution')  # 设置第二个子图的标题
axs[1].set_xlabel('Length')  # 设置X轴标签
# axs[1].set_ylabel('Frequency')  # Y轴标签可以省略，因为和第一个子图共享

axs[2].hist(len_answers, bins='auto')
axs[2].set_title('Answer Length Distribution')  # 设置第三个子图的标题
axs[2].set_xlabel('Length')  # 设置X轴标签
# axs[2].set_ylabel('Frequency')  # Y轴标签可以省略，因为和第一个子图共享

# 设置总图的标题
plt.suptitle('Text Length Distributions')  # 设置整个图形的标题

plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，为总标题留出空间
plt.show()

# 合并字符串
ref_text = ' '.join(references)
ques_text = ' '.join(questions)
ans_text = ' '.join(answers)

# 创建WordCloud实例
custom_stopwords = set(STOPWORDS)
custom_stopwords.update(["yes", "used"])  # 添加更多你想排除的词汇

# 创建WordCloud实例，包含自定义的停用词
wordcloud = WordCloud(stopwords=custom_stopwords, width=800, height=400, background_color ='white')
# 绘制词云
plt.figure(figsize=(20, 10))

# 参考文本的词云
plt.subplot(1, 3, 1)
plt.imshow(wordcloud.generate(ref_text))
plt.title('References Word Cloud')
plt.axis('off')

# 问题文本的词云
plt.subplot(1, 3, 2)
plt.imshow(wordcloud.generate(ques_text))
plt.title('Questions Word Cloud')
plt.axis('off')

# 答案文本的词云
plt.subplot(1, 3, 3)
plt.imshow(wordcloud.generate(ans_text))
plt.title('Answers Word Cloud')
plt.axis('off')

plt.show()  