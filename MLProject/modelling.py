import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

print(f"Tracking URI: {mlflow.get_tracking_uri()}")

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

# MLflow Project sudah membuka run
mlflow.log_metric("accuracy", acc)
mlflow.sklearn.log_model(model, artifact_path="model")