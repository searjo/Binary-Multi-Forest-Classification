#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

LRP = LogisticRegression(penalty='l2', C=1.0, #or C=searchCV.C_[0], 
                           fit_intercept=True, 
                           solver='newton-cg', max_iter=5000)
#Change to newton-cg because liblinear is only used for binary. 
#Newton-cg only works with l2 regularization

"""## Get predictions

### Baseline prediction

Below with no tuning of hyperparameters or polynomial transformations - just a simple logistic regression for baseline
"""

y_pred_baseline, pred_test_baseline = multi_regression(MLR, x_train, ym, test)
evaluate_multi('Basic Multi', MLR, x_train, ym, y_pred_baseline)

"""F-1 Score is not great for any of my Cover_Types, particularly Cover_Type 5 (0.00). A lot of room for improvement."""

#Assess with CVP
baseline_pred_cv = multi_cvp(LRP, x_train, ym, 5)
evaluate_multi('Random Undersampler Multi CVP', LRP, x_train, ym, baseline_pred_cv)

"""### Try RandomUndersampler

Attempt to perform basic undersampler without reweighting to see scores for baseline. Note I eliminated probabilities as they are not of interest for submission, so won't plot the ROC curve. However, classification support will suffice 
"""

#Create new data with y so I can make changes to a new dataset with representative number of y-variables
rus_multi = x_plf.copy()

# Create an instance of RandomUnderSampler, fit the training data and apply Logistics regression
from imblearn.under_sampling import RandomUnderSampler

RUSM = RandomUnderSampler()
rus_multi, y_rusm = RUSM.fit_resample(rus_multi, ym)

y_hat_rusm, pred_test_rusm = multi_regression(LRP, rus_multi, y_rusm, test_plf)
evaluate_multi('Random Undersampler Multi', LRP, rus_multi, y_rusm, y_hat_rusm)

"""F1 Score improved greatly! All numbers are now .7 or higher. Must assess with CVP"""

#Assess accuracy using CVP
rusm_pred_cv = multi_cvp(LRP, rus_multi, y_rusm, 5)
evaluate_multi('Random Undersampler Multi CVP', LRP, rus_multi, y_rusm, rusm_pred_cv)

"""### SMOTE with Polynomials

This method worked best for me last time, so it's worth trying once more
"""

# Create an instance of SMOTE, fit the training data and apply logistic regression
smoter = SMOTE(random_state=59, sampling_strategy='minority', k_neighbors=5)
smote_trainm, y_smotem = smoter.fit_resample(x_plf, ym)

smoter_classifier = LRP

smote_predm, smote_testm = multi_regression(smoter_classifier, smote_trainm, y_smotem, test_plf)
evaluate_multi('SMOTE Multi Poly', LRP, smote_trainm, y_smotem, smote_predm)

"""### Try SMOTETomek"""

st = SMOTETomek(smote=SMOTE(random_state=0), tomek=TomekLinks(sampling_strategy='majority'))
stx, y_st = st.fit_resample(x_plf, ym)

y_hat_st, pred_test_st = multi_regression(LRP, stx, y_st, test_plf)
evaluate_multi('Random Undersampler Multi', LRP, stx, y_st, y_hat_st)

"""### Reweighting

Reweighting will tell the algorithm to give more importance to smaller classes, like the Cover_Type 7 which was targeted in the binary example. I can do this through the class_weight function from sklearn


"""

LR3 = LogisticRegression(penalty='l2', C=1.0, #or C=searchCV.C_[0], 
                           fit_intercept=True, multi_class='multinomial', solver='lbfgs', max_iter=5000)

from  sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes = np.unique(ym), y = np.ravel(ym))

cw = dict(zip(np.unique(ym), class_weights))
cw

"""### Create pipeline and tune hyperparameters

Pipeline allows entire data analysis process to be run as one object
"""

from sklearn.pipeline import Pipeline

pipe = Pipeline([('classifier' , RandomForestClassifier())])
# pipe = Pipeline([('classifier', RandomForestClassifier())])

# Create param grid.

param_grid = [
    {'classifier' : [LogisticRegression()],
     'classifier__penalty' : ['l1', 'l2'],
    'classifier__C' : np.logspace(-4, 4, 20),
    'classifier__solver' : ['liblinear']},
    {'classifier' : [RandomForestClassifier()],
    'classifier__n_estimators' : list(range(10,101,10)),
    'classifier__max_features' : list(range(6,32,5))}
]

# Create grid search object

clf = GridSearchCV(pipe, param_grid = param_grid, cv = 2, verbose=True, n_jobs=-1)

# Fit on data

best_clf = clf.fit(x_plf, ym)

"""### Random Forest

A random forest is one of the more popular machine learning algorithms out there for data sets like this. Setting it up and running it was easy and didn't take much time even on the polynomial transformed data. I used class weights for the model as well.
"""

from sklearn.ensemble import RandomForestClassifier
# Created the model with 300 trees, but it was identical to 100 trees
RF = RandomForestClassifier(n_estimators=300, #default
                            bootstrap = True, #default
                            class_weight = cw)

#Can use warm start to save time

# Fit on training data
RF.fit(x_plf, cover_types)

rf_pred, rf_prob, rf_test = get_predictions(RF, x_plf, test_plf)

evaluate_multi("Random Forest", RF, x_plf, ym, rf_pred)

submission(rf_test, "multi4.csv")

"""Great! The Kaggle Prediction was .89 with RF. Still room for improvement, but not bad"""