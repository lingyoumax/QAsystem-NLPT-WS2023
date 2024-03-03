FROM python:3.8

WORKDIR /app

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
RUN python -m spacy download en_core_web_sm

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]