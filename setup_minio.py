from minio import Minio
import pandas as pd
from sklearn.datasets import load_iris

# MinIO 클라이언트 설정
client = Minio(
    "0.0.0.0:9000",
    access_key="minio",
    secret_key="rootminio",
    secure=False,
)

bucket_name = "raw-data"
object_name = "iris"

# 버킷이 존재하는지 확인하고, 없다면 생성
if not client.bucket_exists(bucket_name):
    print(f"Creating bucket: {bucket_name}")
    client.make_bucket(bucket_name)
else:
    print(f"Bucket {bucket_name} already exists")

# iris 데이터 로드
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# CSV 파일로 저장
df.to_csv("iris.csv", index=False)

# MinIO에 업로드
print(f"Uploading {object_name} to {bucket_name}")
client.fput_object(bucket_name, object_name, "iris.csv")

print("Setup completed successfully!") 