from datetime import datetime

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from joblib import dump
from src.data_processing.process_titanic_data import get_training_data_split

X_train, X_test, y_train, y_test = get_training_data_split(test_size=0.3)

# create model
model = DecisionTreeClassifier()
model = model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # y_proba = probability for class 1 (survivors)

# save model
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
dump(model, f"classification_tree_{timestamp}.joblib")

# metrics
print("=== Classification Report ===")
print(metrics.classification_report(y_test, y_pred))
print("__________")

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 Score:", metrics.f1_score(y_test, y_pred))
print("__________")

print("Log Loss (Entropy):", metrics.log_loss(y_test, y_proba)) # cf. ISLP textbook page 356
print("__________")

print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))

# plot
plt.figure(figsize=(12,8))
plot_tree(model, filled=True, feature_names=X_train.columns, class_names=True)
plt.show()



