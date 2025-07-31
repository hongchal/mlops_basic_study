import mlflow 

mlflow.set_tracking_uri("http://localhost:5001") 
mlflow.set_experiment("test")

for i in range(3):
    with mlflow.start_run():
        mlflow.log_param("trial", i)
        mlflow.log_metric("metric", i)

