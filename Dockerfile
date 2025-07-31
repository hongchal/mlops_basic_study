FROM python:3.10-slim

RUN pip install -U pip &&\
    pip install mlflow

CMD ["mlflow", "server", "--host", "0.0.0.0"]
