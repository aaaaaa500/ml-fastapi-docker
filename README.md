# ML FastAPI Docker

A simple ML model (RandomForest on the Iris dataset) served as an API with FastAPI and packaged with Docker.

## Project structure

- `train.py` - trains the model and saves it to `model.joblib`
- `main.py` - FastAPI app (endpoints `GET /` and `POST /predict`)
- `model.joblib` - saved trained model
- `requirements.txt` - Python dependencies
- `Dockerfile` - container instructions

## Run locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train.py
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs to test the API in Swagger.

## Run with Docker

```bash
docker build -t ml-fastapi-app .
docker run -p 8000:8000 ml-fastapi-app
```

Open http://127.0.0.1:8000/docs

## API

`GET /` returns:

```json
{"message": "ML API is running"}
```

`POST /predict` example request:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Example response:

```json
{
  "prediction": 0,
  "class_name": "setosa"
}
```