FROM python:3.10-slim

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY api api
COPY frontend frontend
COPY model model
COPY preprocessing_package preprocessing_package

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
