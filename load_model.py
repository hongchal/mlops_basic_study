import os
import pandas as pd
from minio import Minio
import mlflow 

BUCKET_NAME = "raw-data"
OBJECT_NAME = "iris"

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://0.0.0.0:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://0.0.0.0:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "rootminio"

def download_data():
    client = Minio(
        "0.0.0.0:9000",
        access_key="minio",
        secret_key="rootminio",
        secure=False,
    )   


    object_stat = client.stat_object(BUCKET_NAME, OBJECT_NAME)
    data_version_id = object_stat.version_id
    client.fget_object(BUCKET_NAME, OBJECT_NAME, "downloaded_iris.csv")

    return data_version_id

def load_data():
    data_version_id = download_data()
    df = pd.read_csv("downloaded_iris.csv")
    X, y = df.drop(columns=["target"]), df["target"]
    data_dict = {"data_version_id": data_version_id, "data": X, "target": y}
    return data_dict

def load_sklearn_model(run_id, model_name): 
    clf = mlflow.sklearn.load_model(f"runs:/{run_id}/{model_name}")
    return clf

def load_pyfunc_model(run_id, model_name):
    clf = mlflow.pyfunc.load_model(f"runs:/{run_id}/{model_name}")
    return clf

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    
    data_dict = load_data()
    X = data_dict["data"]
    
    # sklearn_clf = load_sklearn_model(args.run_id, args.model_name)
    # sklearn_pred = sklearn_clf.predict(X)
    # print(sklearn_clf)
    # print(sklearn_pred)

    pyfunc_clf = load_pyfunc_model(args.run_id, args.model_name)
    pyfunc_pred = pyfunc_clf.predict(X)
    print(pyfunc_clf)
    print(pyfunc_pred)  