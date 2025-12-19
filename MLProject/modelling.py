import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_experiment("ci-mlflow")

train = pd.read_csv("dataset_preprocessing/diabetes_train_preprocessed.csv")
test = pd.read_csv("dataset_preprocessing/diabetes_test_preprocessed.csv")

X_train = train.drop(columns=["diabetes"])
y_train = train["diabetes"]
X_test = test.drop(columns=["diabetes"])
y_test = test["diabetes"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

mlflow.log_metric("accuracy", acc)

os.makedirs("model", exist_ok=True)
mlflow.sklearn.save_model(model, "model")
mlflow.sklearn.log_model(model, artifact_path="model")