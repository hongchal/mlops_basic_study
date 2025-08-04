import mlflow 
import pandas as pd 
from fastapi import FastAPI
from pydantic import BaseModel

model = mlflow.pyfunc.load_model('./downloads/my_model')

class PredictIn(BaseModel):
    sepal_length:float
    sepal_width:float
    petal_length:float
    petal_width:float

class PredictOut(BaseModel):
    iris_class: str 


app = FastAPI()

@app.post("/predict/", response_model = PredictOut)
def predict(predict_in: PredictIn) -> PredictOut:
    df = pd.DataFrame([predict_in.model_dump()]) #sepal_length
    df.columns = df.columns.str.replace("_", " ") #sepal length
    df = df.add_suffix(" (cm)") # sepal length (cm)
    pred = model.predict(df).item()
    return PredictOut(iris_class=pred)