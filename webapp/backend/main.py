import asyncio
from fastapi import FastAPI, BackgroundTasks
from starlette.middleware.cors import CORSMiddleware

from mongodb import question_collection
from QuestionModel import Question, TIME_STAMP_Model
from process import process
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import HTTPException


@app.post("/sendQuestion")
async def root(input: Question):
    try:
        question = input.question
        TIME_STAMP = input.time_stamp
        year = input.year  # "1999-2010"
        author = input.author
        if year == '-':
            year = ""
        else:
            year = year.split('-')


        question_doc = {
            "question": question,
            "year": year,
            "author": author,
            "time_stamp": TIME_STAMP,
            "input": True,
            'retrieval': False,
            'answerGeneration': False,
            'output': False,
            'reference': "",
            'answer': ""
        }
        await question_collection.insert_one(question_doc)


        asyncio.create_task(process(TIME_STAMP, question, year, author))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "question input successful"}


@app.post("/questionStatus")
async def getStatus(input: TIME_STAMP_Model):
    try:
        question_status = await question_collection.find_one({"time_stamp": input.TIME_STAMP})
        if not question_status:
            raise HTTPException(status_code=404, detail="Question not found")

        status = {'input': question_status['input'],
                  'retrieval': question_status['retrieval'],
                  'answerGenerating': question_status['answerGeneration'],
                  'output': question_status['output'],
                  'reference': question_status['reference']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {'res': 'success', 'status': status}


@app.post("/getAnswer")
async def getAnswer(input: TIME_STAMP_Model):
    try:
        question_status = await question_collection.find_one({"time_stamp": input.TIME_STAMP})
        if not question_status:
            raise HTTPException(status_code=404, detail="Question not found or answer not generated yet")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {'res': 'success', 'answer': question_status['answer']}
