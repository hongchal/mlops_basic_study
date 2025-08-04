from datetime import datetime
from sklearn.datasets import load_iris
from minio import Minio 


iris = load_iris(as_frame=True)
X = iris.data
X.sample(100).to_csv("batch.csv", index=False)

url = '0.0.0.0:9000'
access_key = 'minio'
secret_key = 'rootminio'
client = Minio(
    url,
    access_key=access_key,
    secret_key=secret_key,
    secure=False,
)

bucket_name = 'not-predicted'
object_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if not client.bucket_exists(bucket_name):
    client.make_bucket(bucket_name)

client.fput_object(bucket_name, object_name, "batch.csv")

print(f"Batch data uploaded to {bucket_name}/{object_name}")