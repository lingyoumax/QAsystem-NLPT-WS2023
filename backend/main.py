import asyncio
from fastapi import FastAPI
from mongodb import question_collection
from QuestionModel import Question
from process import process

app = FastAPI()


@app.post("/sendQuestion")
async def root(input: Question):
    question = input.question
    TIME_STAMP = input.time_stamp
    year = input.year #"1999-2010"
    # Write to database with question, TIME_STAMP, and input set to true, others set to false
    question_doc = {"question": question, "year": year.split('-'), "time_stamp": TIME_STAMP, "input": True, 'classification': False,
                    'retrieval': False, 'answerGeneration': False, 'output': False, 'reference': "", 'type': "", 'answer': ""}
    question_collection.insert_one(question_doc)
    asyncio.create_task(process(TIME_STAMP, question, year))
    return {"message": "question input successful"}



@app.get("/questionStatus/{TIME_STAMP}")
async def getStatus(TIME_STAMP: str):
    # Query the database for various statuses based on TIME_STAMP and return them
    question_status = await question_collection.find_one({"time_stamp": TIME_STAMP})
    status = {'input': question_status.input,
              'classification': question_status.classification,
              'retrieval': question_status.retrieval,
              'answerGenerating': question_status.answerGenerating,
              'output': question_status.output
              }

    return {'res': 'success', 'status': f'{status}'}
