from joblib import load

from src.data_processing.process_titanic_data import get_evaluation_data

model = load("classification_tree.joblib")

X = get_evaluation_data()
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

print(y_pred)