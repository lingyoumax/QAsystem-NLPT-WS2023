from pydantic import BaseModel


class Question(BaseModel):
    question: str
    time_stamp: str
    year: str