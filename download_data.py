from minio import Minio

client = Minio(
    "0.0.0.0:9000",
    access_key="minio",
    secret_key="rootminio",
    secure=False,
)

bucket_name = "raw-data"
object_name = "iris"

object_stat = client.stat_object(bucket_name, object_name)
print(object_stat.version_id)

client.fget_object(bucket_name, object_name, "downloaded_iris.csv")