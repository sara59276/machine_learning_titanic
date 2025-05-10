from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load

from constants.models import Models
from src.data_processing.input import get_training_data_split, get_evaluation_data
from src.data_processing.output import create_accuracy_file, create_predictions_file

MODEL_NAME = Models.CLASSIFICATION_TREE.value

X_train, X_test, y_train, y_test, _ = get_training_data_split(test_size=0.3, selected_features_only=True)

# create model
model = DecisionTreeClassifier()
model = model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # y_proba = probability to be survivors

# save model
dump(model, f"{MODEL_NAME}_model.joblib")

# metrics
print("=== Classification Report ===")
print(metrics.classification_report(y_test, y_pred))
print()

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
print("Log Loss (Entropy):", log_loss(y_test, y_proba)) # cf. ISLP textbook page 356
print()
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def _load_model(model_name: str):
    try:
        return load(f"{model_name}_model.joblib")
    except FileNotFoundError:
        print(f"The model hasn't been created yet. Please run the create_{model_name}.py file first.")

trained_model = _load_model(MODEL_NAME)
X, ids = get_evaluation_data(selected_features_only=True)
y_pred = trained_model.predict(X)
create_predictions_file(
    model_name=MODEL_NAME,
    passenger_ids=ids,
    predictions=y_pred,
)
