from joblib import load

model = load("classification_tree.joblib")

X = load_titanic_eval()

y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]