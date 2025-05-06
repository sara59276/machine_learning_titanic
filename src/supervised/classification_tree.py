from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree

from src.data_processing.process_titanic_data import load_titanic_split_sets

# Classification tree >< Regression tree
#    On choisit Classification Tree parce que notre target/VD/y est
#    catégorielle/discrète/non-continue (soit on survit, soit on ne survit pas au crash du Titanic)

X_train, X_test, y_train, y_test = load_titanic_split_sets(test_size=0.3)

# model
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

# predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]  # y_proba = probability for class 1 (survivors)

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
plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=True)
plt.show()


