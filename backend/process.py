import torch
from classification.model import BERTMultiLabelBinaryClassification_FactorLoss
from transformers import BertTokenizer
from mongodb import question_collection
from retrieval.retrieval_BM25 import weightBM25
from retrieval.semantic_search import search_arxiv_texts

TYPE = ['Confirmation Questions', 'Factoid-type Questions', 'List-type Questions', 'Causal Questions',
        'Hypothetical Questions', 'Complex Questions']


async def classify(question):
    load_model_name = "./classification/model_state_dict_all_data_1.pth"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTMultiLabelBinaryClassification_FactorLoss(num_labels=6, gamma=2.5,
                                                          mix_ratio=0.3, label_weight=[0, 0, 0, 0, 0, 0])
    model.load_state_dict(torch.load(load_model_name, map_location=device))

    sentence_encodings = tokenizer(question, truncation=True, padding=True, max_length=128)
    sentence_seq = torch.tensor(sentence_encodings['input_ids'])
    sentence_mask = torch.tensor(sentence_encodings['attention_mask'])
    model.eval()
    with torch.no_grad():
        model.cpu()
        inputs = {
            'input_ids': sentence_seq.unsqueeze(0),
            'attention_mask': sentence_mask.unsqueeze(0)
        }
        outputs = model(**inputs)
        logits = torch.sigmoid(outputs).detach().cpu().numpy()

    pred_labels = (logits > 0.5).astype(int)
    return pred_labels[0]


async def retrieval(type, question, year, top_k=1):
    if len(type) == 1:
        if 'Causal Questions' in type or 'Complex Questions' in type:
            res = search_arxiv_texts(question)
        elif 'Confirmation Questions' in type or 'Factoid-type Questions' in type:
            res = weightBM25(question, top_k=top_k)
        else:
            res = search_arxiv_texts(question) + ' ' + weightBM25(question, top_k=top_k)
    else:
        res = search_arxiv_texts(question)

    return res


async def answerGeneration(type, question, reference):
    return 1


async def process(TIME_STAMP, question, year):
    query = {"time_stamp": TIME_STAMP}
    # Classification
    type = await classify(question)
    valid_types = [TYPE[i] for i, val in enumerate(type) if val == 1]
    print(valid_types)

    # Update the question in the database with classification set to true, based on TIME_STAMP
    update = {"$set": {"type": ', '.join(valid_types), 'classification': True}}
    question_collection.find_one_and_update(query, update)

    # Choose the retrieval method based on type
    reference = await retrieval(valid_types, question, year)
    # Update the question in the database with retrieval set to true and store the reference
    update = {"$set": {"retrieval": True, 'reference': reference}}
    question_collection.find_one_and_update(query, update)

    print(reference)
    return

    # Choose the answer generation model based on type
    answer = await answerGeneration(type, question, reference)
    # Update the question in the database with answerGeneration set to true, write the answer and set output to true
    update = {"$set": {'answerGeneration': True, 'answer': answer, 'output': True}}
    question_collection.find_one_and_update(query, update)
