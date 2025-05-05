from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree

from src.data_processing.process_titanic_data import load_titanic, load_titanic_train_test

# Classification tree >< Regression tree
#    On choisit Classification Tree parce que notre target/VD/y est
#    catégorielle/discrète/non-continue (soit on survit, soit on ne survit pas au crash du Titanic)

X_train, X_test, y_train, y_test = load_titanic_train_test(test_size=0.3)

# model
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

# predictions
y_pred = clf.predict(X_test)

# metrics
print("Precision:", metrics.classification_report(y_test, y_pred))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Recall:")
print("F1 score:")

# plot
plt.figure(figsize=(12,8))
plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=True)
plt.show()


