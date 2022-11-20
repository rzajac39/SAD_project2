from sklearn.metrics import *
import pandas as pd 
import matplotlib.pyplot as plt
from typing import Union

def plot_roc_auc(clf, X_test, y_test):
    probas = clf.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test, probas[:,1])
    roc_auc = auc(fpr, tpr)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    return roc_display

def plot_confusion_matrix(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    cm_lr = confusion_matrix(y_test, y_pred)
    return ConfusionMatrixDisplay(cm_lr).plot()

def generate_raport(clf, X_train, X_test, y_train, y_test, get_frame=False) -> Union[None, pd.Series]:
        """
        Function to compute selected metric of the model and
        Needs better documentation
        """

        
        predictions = clf.predict(X_test)
        
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        accuracy = accuracy_score(y_test, predictions)
        # roc_auc = roc_auc_score(y_test, predictions)

        probas = clf.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, probas[:,1])
        roc_auc = auc(fpr, tpr)
        
        
        if get_frame:
            tmp = {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Accuracy": accuracy,
            "Roc_auc_score": roc_auc
            }
            return pd.DataFrame.from_dict(tmp, orient = 'index').T
        
        print('Model test:')
        print(f'\t\tPrec: {precision} \
                \n\t\t Rec: {recall} \
                \n\t\t F1: {f1} \
                \n\t\t Acc: {accuracy} \
                \n\t\t ROC_AUC: {roc_auc} \
                ')
                