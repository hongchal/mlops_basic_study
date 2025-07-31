FROM python:3.10-slim

WORKDIR /usr/app

RUN pip install -U pip &&\
    pip install pandas==2.2.3 scikit-learn==1.7.1

COPY train.py train.py

ENTRYPOINT ["/bin/bash"]