import os 
from minio import Minio
import pandas as pd
import mlflow

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://0.0.0.0:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://0.0.0.0:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "rootminio"

def predict(run_id, model_name):
    clf = mlflow.pyfunc.load_model(f"runs:/{run_id}/{model_name}") 

    url = '0.0.0.0:9000'
    access_key = 'minio'
    secret_key = 'rootminio'
    client = Minio(
        url,
        access_key=access_key,
        secret_key=secret_key,
        secure=False,
    )

    if "predicted" not in client.list_buckets():
        client.make_bucket("predicted")

    not_predicted_objects = [object.object_name for object in client.list_objects(bucket_name="not-predicted")]
    predicted_objects = [object.object_name for object in client.list_objects(bucket_name="predicted")]

    to_predict = []

    for object_name in not_predicted_objects:
        if object_name not in predicted_objects:
            to_predict.append(object_name)
    
    for file_name in to_predict:
        print("data to predict: ", file_name)

        client.fget_object(bucket_name="not-predicted", object_name=file_name, file_path=file_name)
        data_df = pd.read_csv(file_name)

        predictions = clf.predict(data_df)
        
        pred_file_name = f"pred_{file_name}"
        predictions.to_csv(pred_file_name, index=False)

        client.fput_object(bucket_name="predicted", object_name=file_name, file_path=pred_file_name)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    
    predict(args.run_id, args.model_name)