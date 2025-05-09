import pandas as pd


def create_csv(filename: str, passenger_id, survived):
    output_df = pd.DataFrame({
        "PassengerId": passenger_id,
        "Survived": survived,
    })

    output_df.to_csv("classification_tree.csv", index=False)