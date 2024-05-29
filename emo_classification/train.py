from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from utils import split_for_train, load_train_dev
import numpy as np
from sklearn.linear_model import RidgeClassifier



train_df, eval_df, _= split_for_train()

X_train = np.load("train_embeddings_subset_hadqaet.npy")
y_train = train_df.labels.values.codes

X_eval = np.load("eval_embeddings_subset_hadqaet.npy")
y_eval = eval_df.labels.values.codes

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=500)

# SVM Classifier
svm_classifier = SVC(kernel="rbf")

ridge_classifier = RidgeClassifier()


# Specify the number of folds for cross-validation
n_folds = 3
rf_accuracies = []
svm_accuracies = []
ridge_accuracies = []
rf_f1s = []
svm_f1s = []
ridge_f1s = []
rf_precisions = []
svm_precisions = []
ridge_precisions = []
rf_recalls = []
svm_recalls = []
ridge_recalls = []

for i in range(n_folds):
    
    print("Fold number: ", i+1)
    print("Training...")
    rf_classifier.fit(X_train[:,0,:].squeeze(), y_train)
    svm_classifier.fit(X_train[:,0,:].squeeze(), y_train)
    ridge_classifier.fit(X_train[:,0,:].squeeze(), y_train)

    print("Evaluating...")
    rf_predictions = rf_classifier.predict(X_eval[:,0,:].squeeze())
    svm_predictions = svm_classifier.predict(X_eval[:,0,:].squeeze())
    ridge_predictions = ridge_classifier.predict(X_eval[:,0,:].squeeze())

    rf_accuracy = accuracy_score(y_eval, rf_predictions)
    svm_accuracy = accuracy_score(y_eval, svm_predictions)
    ridge_accuracy = accuracy_score(y_eval, ridge_predictions)

    rf_f1 = f1_score(y_eval, rf_predictions, average='macro')
    svm_f1 = f1_score(y_eval, svm_predictions, average='macro')
    ridge_f1 = f1_score(y_eval, ridge_predictions, average='macro')

    rf_precision = precision_score(y_eval, rf_predictions, average='macro')
    svm_precision = precision_score(y_eval, svm_predictions, average='macro')
    ridge_precision = precision_score(y_eval, ridge_predictions, average='macro')

    rf_recall = recall_score(y_eval, rf_predictions, average='macro')
    svm_recall = recall_score(y_eval, svm_predictions, average='macro')
    ridge_recall = recall_score(y_eval, ridge_predictions, average='macro')

    rf_accuracies.append(rf_accuracy)
    svm_accuracies.append(svm_accuracy)
    ridge_accuracies.append(ridge_accuracy)
    rf_f1s.append(rf_f1)
    svm_f1s.append(svm_f1)
    ridge_f1s.append(ridge_f1)
    rf_precisions.append(rf_precision)
    svm_precisions.append(svm_precision)
    ridge_precisions.append(ridge_precision)
    rf_recalls.append(rf_recall)
    svm_recalls.append(svm_recall)
    ridge_recalls.append(ridge_recall)
    print("Fold Complete!")

print("Accuracy of random forest classifier: ", np.mean(rf_accuracies))
print("Accuracy of SVM classifier: ", np.mean(svm_accuracies))
print("Accuracy of ridge classifier: ", np.mean(ridge_accuracies))
print("F1 score of random forest classifier: ", np.mean(rf_f1s))
print("F1 score of SVM classifier: ", np.mean(svm_f1s))
print("F1 score of ridge classifier: ", np.mean(ridge_f1s))
print("Precision score of random forest classifier: ", np.mean(rf_precisions))
print("Precision score of SVM classifier: ", np.mean(svm_precisions))
print("Precision score of ridge classifier: ", np.mean(ridge_precisions))
print("Recall score of random forest classifier: ", np.mean(rf_recalls))
print("Recall score of SVM classifier: ", np.mean(svm_recalls))
print("Recall score of ridge classifier: ", np.mean(ridge_recalls))

