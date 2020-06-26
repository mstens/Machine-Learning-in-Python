import pandas as pd
from sklearn.datasets import fetch_openml
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# mnist = fetch_openml("mnist_784", version=1)
# joblib.dump(mnist, "mnist.pkl")
mnist = joblib.load("mnist.pkl")
X, y = mnist["data"], mnist["target"]
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image, cmap="binary")
# plt.show()
# plt.close()
# print(y[0])
y = y.astype(np.uint8)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train, y_train_5)
# joblib.dump(sgd_clf, "sgd_clf.pkl")
sgd_clf = joblib.load("sgd_clf.pkl")
# print(sgd_clf.predict([some_digit]))
# print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
print(confusion_matrix(y_train_5, y_train_pred))
print("precision:", precision_score(y_train_5, y_train_pred), "\nrecall:", recall_score(y_train_5, y_train_pred))
print(f1_score(y_train_5, y_train_pred))

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.legend()


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
plt.close()

threshold_90_precision = thresholds[np.argmax(precisions >= 0.9)]
y_train_pred_90 = (y_scores >= threshold_90_precision)
print(precision_score(y_train_5, y_train_pred_90))
print(recall_score(y_train_5, y_train_pred_90))

from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr, label=""):
    plt.plot(fpr, tpr, "b-", label=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.legend()


plot_roc_curve(fpr, tpr)
plt.show()
plt.close()

print(roc_auc_score(y_train_5, y_scores))

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_probas_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, label="Random forest")
plt.legend(loc="lower right")
plt.show()
plt.close()
print(roc_auc_score(y_train_5, y_probas_forest))
