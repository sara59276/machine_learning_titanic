import statsmodels.api as sm
from joblib import dump, load
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from constants.models import Models
from src.data_processing.input import get_training_data_split, get_evaluation_data
from src.data_processing.output import create_predictions_file, create_accuracy_file


MODEL_NAME = Models.LOGISTIC_REGRESSION.value

X_train, X_test, y_train, y_test, _ = get_training_data_split(test_size=0.3, selected_features_only=True)
X_train = sm.add_constant(X_train) # adding an intercept # TODO : why important ?
X_test = sm.add_constant(X_test)

model = sm.GLM(
    y_train,
    X_train,
    family=sm.families.Binomial(), # TODO justify binomial (target is binary)
)
model = model.fit()
print(model.summary())


y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int) # pour rendre binaire (0 et 1)

dump(model, f"{MODEL_NAME}_model.joblib")

accuracy = accuracy_score(y_test, y_pred)
accuracy = round(accuracy, 4)
create_accuracy_file(
    model_name=MODEL_NAME,
    accuracy=accuracy,
)

print("Accuracy:", accuracy)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

def _load_model(model_name: str):
    try:
        return load(f"{model_name}_model.joblib")
    except FileNotFoundError:
        print(f"The model hasn't been created yet. Please run the create_{model_name}.py file first.")

trained_model = _load_model(MODEL_NAME)
X, ids = get_evaluation_data(selected_features_only=True)
X = sm.add_constant(X)
y_pred = trained_model.predict(X)
y_pred = (y_pred > 0.5).astype(int) # pour rendre binaire (0 et 1)

create_predictions_file(
    model_name=MODEL_NAME,
    passenger_ids=ids,
    predictions=y_pred,
)