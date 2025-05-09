from joblib import load

from src.data_processing.input import get_evaluation_data
from src.data_processing.output import create_csv

try:
    trained_model = load("classification_tree.joblib")

    X, ids = get_evaluation_data()
    y_pred = trained_model.predict(X)

    create_csv(
        filename="classification_tree.csv",
        passenger_id=ids,
        survived=y_pred,
    )

except FileNotFoundError:
    print("The model hasn't been created yet. Please run the create file first.")




