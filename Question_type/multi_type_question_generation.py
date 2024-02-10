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


def generate_questions_type():
    prompt = '''
There are 6 type of questions:
Confirmation Questions: These are queries that can be answered with a simple "yes" or "no," such as "Is Paris the capital of France?"
Factoid-type Questions: These questions typically start with words like "what," "which," "when," "who," or "how," and they seek factual information, such as "What is the population of Paris?"
List-type Questions: These questions expect a list of items as an answer. For instance, "List five major cities in France."
Causal Questions: Causal questions aim to understand the reasons or explanations behind events or phenomena, such as "Why did the French Revolution happen?"
Hypothetical Questions: These questions explore hypothetical scenarios and often begin with "what would happen if." An example is, "What would occur if France never experienced rain?"
Complex Questions: Complex questions require multi-step reasoning and the integration of information from various sources. For example, "Explain the historical, cultural, and economic factors that led to the construction of the Eiffel Tower."

I need you help me generate 5 questions, each question belongs to at least two types from above, which means each question you generated
belongs to two types (For example Confirmation Questions and List-type Questions). Careful: you may let the question belongs to two or three
types randomly. 

The answer list is like the format below:
[Question, Confirmation, Factoid, List, Causal, Hypothetical, Complex]
Question is the question you generate and if this question belongs to Confirmation Questions then Confirmation is 1 otherwise it's 0.

Format:
[question you generate, 0, 1, 0, 1, 0, 0];[question you generate, 1, 0, 0, 1, 0, 0];[question you generate, 0, 0, 1, 1, 0, 0];[question you generate, 0, 0, 0, 1, 0, 1];[question you generate, 0, 1, 0, 0, 1, 0];
the output must be like Format above
'''

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=[
            {"role": "system", "content": prompt}
        ],
        seed=123456
    )
    return response['choices'][0]['message']['content']


if __name__ == "__main__":
    column_names = ['Question', 'Confirmation', 'Factoid', 'List', 'Causal', 'Hypothetical', 'Complex']
    csv_filename = 'multi_type_question.csv'
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)
        for i in tqdm(range(50)):
            data = generate_questions_type().split('\n')
            res = []
            for i in range(len(data)):
                line = data[i]
                line = line[3:]
                temp = line.split(' [')
                if temp[0][0] == '"':
                    temp[0] = temp[0][1:]
                    temp[0] = temp[0][:-1]

                temp[1] = temp[1][:-1]
                temp[1] = temp[1].split(', ')
                temp[1].insert(0, temp[0])
                writer.writerow(temp[1])



    print(f'{csv_filename} has been created with the provided data.')
