import pandas as pd

EXPECTED_SIZE = 418


def create_predictions_file(model_name, passenger_ids, predictions):
    if (len(passenger_ids) != EXPECTED_SIZE
            or len(predictions) != EXPECTED_SIZE):
        raise Exception(f"Number of lines in both columns must be strictly equal to {EXPECTED_SIZE}.\n"
                        f"\tPassengerId: {len(passenger_ids)}\n"
                        f"\tSurvived: {len(predictions)}")

    df = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions,
    })

    df.to_csv(f"{model_name}_predictions.csv", index=False)
