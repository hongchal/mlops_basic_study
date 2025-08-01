import os
import mlflow
import optuna
import uuid
import pandas as pd
from minio import Minio
import dill
import textwrap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

UNIQUE_PREFIX = str(uuid.uuid4())[:8]
BUCKET_NAME = "raw-data"
OBJECT_NAME = "iris"

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://0.0.0.0:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://0.0.0.0:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "rootminio"

class MyModel:
    def __init__(self, clf):
        self.clf = clf
    
    def predict(self, X):
        X_pred = self.clf.predict(X)
        X_pred_df = pd.Series(X_pred).map(
            {
                0: "virginica",
                1: "setosa",
                2: "versicolor"
            },
        )
        return X_pred_df

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

def objective(trial):
    trial.suggest_int("n_estimators", 100, 1000, step=100)
    trial.suggest_int("max_depth", 3, 10)

    run_name = f"{UNIQUE_PREFIX}_trial_{trial.number}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(trial.params)

        data_dict = load_data()
        mlflow.log_param("bucket_name", BUCKET_NAME)
        mlflow.log_param("object_name", OBJECT_NAME)
        mlflow.log_param("data_version_id", data_dict["data_version_id"])
        
        X, y = data_dict["data"], data_dict["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        rf_classifier = RandomForestClassifier(n_estimators=trial.params["n_estimators"], max_depth=trial.params["max_depth"], random_state=42)
        rf_classifier.fit(X_train, y_train)

        y_pred = rf_classifier.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc_score)
        return acc_score

def train_best_model(params):
    run_name = f"{UNIQUE_PREFIX}_best_model"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        
        data_dict = load_data()
        mlflow.log_param("bucket_name", BUCKET_NAME)
        mlflow.log_param("object_name", OBJECT_NAME)
        mlflow.log_param("data_version_id", data_dict["data_version_id"])

        X, y = data_dict["data"], data_dict["target"]
        rf_classifier = RandomForestClassifier(**params, random_state=42)
        rf_classifier.fit(X, y)

        my_model = MyModel(rf_classifier)

        with open("model.dill", "wb") as f:
            dill.dump(my_model, f)

        with open("loader.py", "w") as f:
            f.write(textwrap.dedent(
"""
import os 
import dill 

def _load_pyfunc(path):
    if os.path.isdir(path):
        path = os.path.join(path, "model.dill") 

    with open(path, "rb") as f:
        return dill.load(f)
"""
            ))

        mlflow.pyfunc.log_model( 
            name="my_model",
            data_path="model.dill",
            loader_module="loader"
        )

        return rf_classifier


if __name__ == "__main__":

    client = Minio(
        "0.0.0.0:9000",
        access_key="minio",
        secret_key="rootminio",
        secure=False,
    )

    if not client.bucket_exists("mlflow"):
        client.make_bucket("mlflow")
        print("Created 'mlflow' bucket for MLflow artifacts")

    study_name = "hpo_tutorial"
        
    mlflow.set_experiment(study_name)

    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name, load_if_exists=True)
    
    study.optimize(objective, n_trials=10)

    print(f"Best params: {study.best_params}")
    print(f"Best value: {study.best_value}")

    best_params = study.best_params
    best_model = train_best_model(best_params)

