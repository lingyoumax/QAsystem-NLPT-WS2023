import csv
import random

import pandas as pd

import openai
import string
import pandas as pd
import matplotlib.pyplot as plt
import math

from tqdm import tqdm

openai.api_key = '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'


def generate_questions_type(array):
    prompt = '''
you can assist in random content generating questions. The array consists of six categories, each representing a specific type of question:
Confirmation Questions: These are queries that can be answered with a simple "yes" or "no," such as "Is Paris the capital of France?"
Factoid-type Questions: These questions typically start with words like "what," "which," "when," "who," or "how," and they seek factual information, such as "What is the population of Paris?"
List-type Questions: These questions expect a list of items as an answer. For instance, "List five major cities in France."
Causal Questions: Causal questions aim to understand the reasons or explanations behind events or phenomena, such as "Why did the French Revolution happen?"
Hypothetical Questions: These questions explore hypothetical scenarios and often begin with "what would happen if." An example is, "What would occur if France never experienced rain?"
Complex Questions: Complex questions require multi-step reasoning and the integration of information from various sources. For example, "Explain the historical, cultural, and economic factors that led to the construction of the Eiffel Tower."
The answer list is like the format below:
[Question, Confirmation, Factoid, List, Causal, Hypothetical, Complex]
For example:
["What is the capital of France?", 0, 1, 0, 0, 0, 0]
["How did the French Revolution start?", 0, 0, 0, 1, 0, 0]
["List three famous French artists.", 0, 0, 1, 0, 0, 0]

In this format, "Question" represents the generated question, and each subsequent value (0 or 1) indicates the question type it belongs to.
So the pipeline is:
1.Generate questions beased on {:}
2.combine question and the row like the example
3.combine all the answer as a list and return.
You should use the pipeline to generate random content question list,and just return list.
'''.format(array)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=[
            {"role": "system", "content": prompt}
        ],
        seed=123456
    )
    return response['choices'][0]['message']['content']


def detect_questions_type(questions):
    prompt = '''
You are a question type detection helper.

I will input a series of questions:{:}, and I need your assistance to classify them into one or more of the following six types:
1. Confirmation Questions: These are yes/no questions that require a binary answer, such as "Is Paris the capital of France?"
2. Factoid-type Questions: These questions typically begin with "what," "which," "when," "who," or "how," and they seek factual information, like "What is the population of Paris?"
3. List-type Questions: The answer to these questions is a list of items. For example, "List five major cities in France."
4. Causal Questions: Causal questions aim to understand the reasons or explanations behind events or phenomena, e.g., "Why did the French Revolution occur?"
5. Hypothetical Questions: These questions explore hypothetical scenarios and often start with "what would happen if." An example is, "What would happen if it never rained in France?"
6. Complex Questions: Complex questions require multi-step reasoning and the integration of information from various sources. For instance, "Explain the historical, cultural, and economic factors that led to the construction of the Eiffel Tower."

The pipeline for classification is as follows:
A. First, detect the type(s) of each question.
B. Finally, return the results in the following format:

[Question, Confirmation, Factoid, List, Causal, Hypothetical, Complex]

For example:
["What is the capital of France?", 0, 1, 0, 0, 0, 0]
["How did the French Revolution start?", 0, 0, 0, 1, 0, 0]
["List three famous French artists.", 0, 0, 1, 0, 0, 0]

In this format, "Question" represents the input question, and each subsequent value (0 or 1) indicates whether the question belongs to a particular type. Please assist me in classifying these questions accurately and only return result.
'''.format(questions)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=[
            {"role": "system", "content": prompt}
        ],
        seed=12
    )
    return response['choices'][0]['message']['content']


import json
import random

if __name__ == "__main__":
    text_type = 2
    num = 500
    savefile = "squad_questions_and_types_balanced_{:}_2".format(num)
    if text_type == 0:
        with open('squad_data.json', 'r', encoding='utf-8') as file:
            squad_data = json.load(file)
        questions_list = []
        for article in squad_data['data']:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    questions_list.append(qa['question'])
        if len(questions_list) >= num:
            selected_questions = random.sample(questions_list, 1000)
        else:
            selected_questions = questions_list
        csv_filename = "{:}.csv".format(savefile)

        with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Question', 'Confirmation', 'Factoid', 'List', 'Causal', 'Hypothetical', 'Complex'])
            for i in tqdm(range(int(num / 5))):
                questionstr = ''
                for j in range(4):
                    questionstr += '"' + selected_questions[i * 5 + j] + '"' + ","
                questionstr += '"' + selected_questions[i * 5 + 4] + '"' + "]"
                res = detect_questions_type(questionstr)

                res = res.split('\n')
                question_result = []
                type_result = []
                for i in res:
                    if "[" in i:
                        temp = i.split("[")[1].split("]")[0]
                        temp = temp.split(",")
                        if len(temp) > 7:
                            for j in temp[1:len(temp) - 6]:
                                temp[0] += j
                            del temp[1:len(temp) - 6]
                        question_result.append(temp[0].replace('"', ""))
                        type_result.append([int(j) for j in temp[1:]])
                if len(question_result) != len(type_result):
                    continue
                for i in range(len(question_result)):
                    csv_writer.writerow(
                        [question_result[i], type_result[i][0], type_result[i][1], type_result[i][2], type_result[i][3],
                         type_result[i][4], type_result[i][5]])
    if text_type == 1:
        csv_filename = "{:}.csv".format(savefile)
        result = []
        for j in [0, 2, 3, 4, 5]:
            for i in tqdm(range(int(num / 5))):
                array = [[1 if idx == j else 0 for idx in range(6)] for t in range(5)]
                print(array)
                res = generate_questions_type(array)
                temp = res.split("\n")
                res = [j for j in temp if "]" in j]
    if text_type == 2:
        tag_list = ["what", "which", "when", "who", "how", "where"]
        with open('squad_data.json', 'r', encoding='utf-8') as file:
            squad_data = json.load(file)
        factoid_list = []
        other_list = []
        for article in squad_data['data']:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    tag = 0
                    for t in tag_list:
                        if t in qa['question'].lower():
                            tag = 1
                            break
                    if tag == 0:
                        other_list.append(qa['question'])
                    else:
                        factoid_list.append(qa['question'])
        if len(other_list) >= 4 * num:
            selected_questions = random.sample(other_list, 4 * num)
        else:
            selected_questions = other_list
        # if len(other_list) >=7*num:
        #     selected_questions.extend(random.sample(other_list, 7*num))
        #     a=0
        # else:
        #     selected_questions.extend(other_list)
        csv_filename = "{:}.csv".format(savefile)

        with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Question', 'Confirmation', 'Factoid', 'List', 'Causal', 'Hypothetical', 'Complex'])
            for i in tqdm(range(int(len(selected_questions) / 5))):
                questionstr = ''
                for j in range(4):
                    if i * 5 + j > len(selected_questions):
                        break
                    questionstr += '"' + selected_questions[i * 5 + j] + '"' + ","
                questionstr += '"' + selected_questions[i * 5 + 4] + '"' + "]"
                res = detect_questions_type(questionstr)

                res = res.split('\n')
                question_result = []
                type_result = []
                for i in res:
                    if "[" in i:
                        temp = i.split("[")[1].split("]")[0]
                        temp = temp.split(",")
                        if len(temp) > 7:
                            for j in temp[1:len(temp) - 6]:
                                temp[0] += j
                            del temp[1:len(temp) - 6]
                        question_result.append(temp[0].replace('"', ""))
                        type_result.append([int(j) for j in temp[1:]])
                if len(question_result) != len(type_result):
                    continue
                for i in range(len(question_result)):
                    csv_writer.writerow(
                        [question_result[i], type_result[i][0], type_result[i][1], type_result[i][2], type_result[i][3],
                         type_result[i][4], type_result[i][5]])

    print(f"数据已写入到'{csv_filename}'")
