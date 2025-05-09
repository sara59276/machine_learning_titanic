import pandas as pd
from joblib import load

from src.data_processing.process_titanic_data import get_evaluation_data

trained_model = load("classification_tree.joblib")

X, ids = get_evaluation_data()
y_pred = trained_model.predict(X)

output_df = pd.DataFrame({
    "PassengerId": ids,
    "Survived": y_pred
})

output_df.to_csv("classification_tree.csv", index=False)