import joblib  
from fastapi import FastAPI  
from pydantic import BaseModel  
app = FastAPI()
model = joblib.load("model.joblib")

class_names = ["setosa", "versicolor", "virginica"]


class IrisFeatures(BaseModel):
    sepal_length: float 
    sepal_width: float 
    petal_length: float  
    petal_width: float  


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(features: IrisFeatures):
    data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]]

    prediction = model.predict(data)

    class_id = int(prediction[0])

    return {
        "prediction": class_id,
        "class_name": class_names[class_id],
    }