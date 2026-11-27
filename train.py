import joblib 
import mlflow  
import mlflow.sklearn 
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("iris-classification")

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}

with mlflow.start_run():
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("Accuracy:", accuracy)
    print("F1-score:", f1)

    mlflow.log_params(params)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    joblib.dump(model, "model.joblib")
    mlflow.log_artifact("model.joblib")

    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name="iris-classifier",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

print("Done: model saved to model.joblib and registered in MLflow")