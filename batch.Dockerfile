FROM python:3.9-slim

WORKDIR /usr/app/

RUN pip install -U pip &&\
    pip install mlflow==3.1.4 boto3==1.40.1 minio==7.2.16

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt 

COPY image_predict.py image_predict.py
COPY loader.py loader.py
ENTRYPOINT [ "python", "image_predict.py"]