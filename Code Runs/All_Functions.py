#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 23:57:09 2021

@author: johnsearight
"""

#Import relevant modules
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

import sklearn
from sklearn.linear_model import LogisticRegression, lars_path
from sklearn.preprocessing import StandardScaler, PolynomialFeatures as plf
from sklearn.model_selection import GridSearchCV, cross_val_predict as cvp
from sklearn.utils import class_weight
from sklearn.datasets import make_classification
from sklearn.metrics import mean_squared_error, make_scorer, confusion_matrix, classification_report, roc_curve, roc_auc_score, PrecisionRecallDisplay

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC

from imblearn.pipeline import make_pipeline
from imblearn.base import BaseSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import (SMOTE, RandomOverSampler)
from imblearn.combine import SMOTEENN, SMOTETomek

"""#Compile Functions"""

#My functions

#define an evaluate function which I will use in future regressions
def binary_regression(regression, x, y, test):
    regression.fit(x, y)
    print("Intercept is:", regression.intercept_)
    y_hat = regression.predict(x)
    #Turn back into df
    y_hat = pd.DataFrame(y_hat, columns = ['Y Prediction'])
    #Get probability, one for both 0 and 1, and one for 1
    #Use [:,1] above to only return probability of one
    y_hat_prob = regression.predict_proba(x)
    y_hat_prob1 = regression.predict_proba(x)[:,1]

    #Turn back into df and print output for prediction and probability 
    y_hat_prob1 = pd.DataFrame(y_hat_prob1, columns = ['Predicted Prob of Y=1'])
    output = pd.concat([y_hat, y_hat_prob1], axis=1)
    print("TRAIN PREDICTIONS")
    print(output)

    #Do same for testing set predictions
    print("TEST PREDICTIONS")
    pred_test = regression.predict(test)
    pred_test = pd.DataFrame(pred_test, columns = ['Y Prediction Test'])
    pred_test_prob = regression.predict_proba(test)[:,1]
    pred_test_prob = pd.DataFrame(pred_test_prob, columns = ['Predicted Prob of Y=1 (Test)'])
    output_test = pd.concat([pred_test, pred_test_prob], axis=1)
    print(output_test)

    return y_hat, y_hat_prob, pred_test, pred_test_prob

def feature_impact(x, regression):
  feature_impact = pd.DataFrame({"Feature":x.columns.tolist(),"Coefficients":regression.coef_[0]})
  print(feature_impact.plot.bar(x='Feature', y='Coefficients', color='green'))
  return feature_impact



#From helper functions for ROC and AUC score
def get_auc(y, y_pred_probabilities, class_labels, column =1, plot = True):
    """Plots ROC AUC
    """
    fpr, tpr, _ = roc_curve(y == column, y_pred_probabilities[:,column],drop_intermediate = False)
    roc_auc = roc_auc_score(y_true=y, y_score=y_pred_probabilities[:,1])
    print ("AUC: ", roc_auc)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()   

def evaluate(title, regression, X_train, y, y_pred, y_pred_prob):
    #title is a string denoting the type of data being used (i.e. undersample, oversample)
    regression_acc = regression.score(X_train, y_pred)
    print("Accuracy ({}): {:.2f}%".format(title, regression_acc * 100))
    #regression_acc * 100 because want to see as percentage
    y_pred = regression.predict(X_train)

    #from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, y_pred)
    clr = classification_report(y, y_pred)
    #classification report is also useful to compare
    
    #plot confusion matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Greens')
    #fmt allows strings 
    #vmin sets minimum value of colors to zero
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    #plot confusion matrix with percentages
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm/np.sum(cm), annot=True,fmt='.2%', cmap='Greens')    
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix with Percentages")
    plt.show()
    

    print("Classification Report:\n----------------------\n", clr)

    get_auc(y, y_pred_prob, ["Not Target", "Target"], column=1, plot=True) # Help function

    #Precision recall curve, which is appropriate for an imbalanced data set.
    #from sklearn.metrics import PrecisionRecallDisplay

    display = PrecisionRecallDisplay.from_estimator(
      regression, X_train, y, name="LinearSVC"
      ) 
    _ = display.ax_.set_title("2-class Precision-Recall curve")


def scale_min2dum(train, test, dummycolumn1, dummycolumn2):
  #from sklearn.preprocessing import StandardScaler
  #remove dummies as they don't need to be scaled
  futuredummy_train = pd.concat([train.pop(x) for x in [dummycolumn1, dummycolumn2]], axis=1)
  futuredummy_test = pd.concat([test.pop(x) for x in [dummycolumn1, dummycolumn2]], axis=1)
  #fit scaler on training set
  scaler = StandardScaler()
  scaler.fit(train)
  train = pd.DataFrame(scaler.transform(train), columns=train.columns)
  #ensure that scaler for training set is applied to test set
  test = pd.DataFrame(scaler.transform(test), columns=test.columns)
  #re-insert future dummies
  train = pd.concat([train, futuredummy_train], axis=1)
  test = pd.concat([test, futuredummy_test], axis=1)
  return train, test

#function that marks insignificant dummies as other
def insignificant_dummies_other(dummy_col, threshold):

    # removes the bind
    dummy_col = dummy_col.copy()

    # what is the ratio of a dummy in whole column
    count = pd.value_counts(dummy_col) / len(dummy_col)

    # cond whether the ratios are higher than the threshold
    mask = dummy_col.isin(count[count > threshold].index)

    # replace the ones which ratio is lower than the threshold by a special name
    dummy_col[~mask] = "Other"

    return pd.get_dummies(dummy_col, prefix=dummy_col.name)

#from sklearn.model_selection import cross_val_predict as cvp

def binary_cvp(regression, x, y, folds):
  from sklearn.model_selection import cross_val_predict as cvp
  y_pred_cv = cvp(regression, x, y, cv=folds)
  y_prob_cv = cvp(regression, x, y, cv=folds, method='predict_proba')
  #Turn back into df
  y_pred_cv = pd.DataFrame(y_pred_cv, columns = ['Y Prediction'])
  #Get probability, one for both 0 and 1, and one for 1
  #Use [:,1] above to only return probability of one
  #Need y_prob_cv for my evaluation
  y_prob_cv = regression.predict_proba(x)
  y_prob1_cv = regression.predict_proba(x)[:,1]
  #Turn back into df and print output for prediction and probability 
  y_prob1_cv = pd.DataFrame(y_prob1_cv, columns = ['Predicted Prob of Y=1'])
  output = pd.concat([y_pred_cv, y_prob1_cv], axis=1)
  print("TRAIN CVP PREDICTIONS")
  print(output)
  cmcvp =  confusion_matrix(y_pred=y_pred_cv, y_true=y, labels=[0,1])
  print ("Confusion Matrix", cmcvp)
  return y_pred_cv, y_prob_cv 

#Initial LR for initial regression
LR = LogisticRegression(penalty='l2', C=100.0, 
                           fit_intercept=True, 
                           intercept_scaling=1, 
                           solver='liblinear', max_iter=500)

#Create function for submitting to Kaggle
def submission(predictions, submission_filename):
  kaggleattempt = predictions.copy()
  kaggleattempt.index +=61006
  kaggleattempt.rename(columns={ kaggleattempt.columns[0]: 'Cover_Type' }, inplace = True)
  return kaggleattempt.to_csv(submission_filename, index=True, index_label='Index')

def separate_multi(train, column):
  y = train[column]
  x = train.drop([column], axis=1)
  return x, y

def multi_regression(regression, x, y, test):
    regression.fit(x, y)
    #print("Intercept is:", regression.intercept_)
    y_hat = regression.predict(x)
    #Turn back into df
    y_hat = pd.DataFrame(y_hat, columns = ['Y Prediction'])

    #Turn back into df and print output for prediction and probability 
    print("TRAIN PREDICTIONS")
    print(y_hat)

    #Do same for testing set predictions
    pred_test = regression.predict(test)
    pred_test = pd.DataFrame(pred_test, columns = ['Y Prediction Test'])
    print("TEST PREDICTIONS")
    print(pred_test)
    return y_hat, pred_test

def get_predictions(model, x, test):
  y_pred = model.predict(x)
  y_pred_prob = model.predict_proba(x)
  y_pred_test = model.predict(test)
  print("Train Predictions:", y_pred)
  print("Test Predictions:", y_pred_test)
  y_pred = pd.DataFrame(y_pred, columns = ['Y Prediction'])
  y_pred_test = pd.DataFrame(y_pred_test, columns = ['Y Test Prediction'])
  return y_pred, y_pred_prob, y_pred_test


def feature_impact(x, regression):
  feature_impact = pd.DataFrame({"Feature":x.columns.tolist(),"Coefficients":regression.coef_[0]})
  print(feature_impact.plot.bar(x='Feature', y='Coefficients', color='green'))
  return feature_impact

def multi_cvp(regression, x, y, folds):
  from sklearn.model_selection import cross_val_predict as cvp
  y_pred_cv = cvp(regression, x, y, cv=folds)
  #Turn back into df
  y_pred_cv = pd.DataFrame(y_pred_cv, columns = ['Y Prediction'])
  #Turn back into df and print output for prediction 
  print("TRAIN CVP PREDICTIONS", y_pred_cv)
  cmcvp =  confusion_matrix(y_pred=y_pred_cv, y_true=y)
  print ("Confusion Matrix", cmcvp)

  return y_pred_cv

#Evaluation function for multi-class. 

def evaluate_multi(title, regression, X_train, y, y_pred):
    #title is a string denoting the type of data being used (i.e. undersample, oversample)
    regression_acc = regression.score(X_train, y_pred)
    print("Accuracy ({}): {:.2f}%".format(title, regression_acc * 100))
    #regression_acc * 100 because want to see as percentage
    y_pred = regression.predict(X_train)

    #from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y, y_pred)
    clr = classification_report(y, y_pred)
    #classification report is also useful to compare
    
    #plot confusion matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Greens')
    #fmt allows strings 
    #vmin sets minimum value of colors to zero
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    #plot confusion matrix with percentages
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm/np.sum(cm), annot=True,fmt='.2%', cmap='Greens')    
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix with Percentages")
    plt.show()

    print("Classification Report:\n----------------------\n", clr)

LRP = LogisticRegression(penalty='l2', C=1.0, #or C=searchCV.C_[0], 
                           fit_intercept=True, 
                           solver='newton-cg', max_iter=5000)
