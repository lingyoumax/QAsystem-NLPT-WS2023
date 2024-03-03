from pydantic import BaseModel


class Question(BaseModel):
    question: str
    time_stamp: str
    year: str
    author: str


class TIME_STAMP_Model(BaseModel):
    TIME_STAMP: str