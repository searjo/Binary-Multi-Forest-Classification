#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""# Get Predictions

## Tune hyperparameters

To find the best parameters for my regression, I can use GridsearchCV or LogisticRegressionCV. I've chosen to run both to see what my best options are, as they can give different reults

#### GridSearchCV

GridSearchCV is way to tune the hyperparameters. You are able to try out different combinations of values for hyperparameters and it will be able to output the best one. GridSearchCV uses cross-validation. Cross-validation, which is used as well to assess model accuracy, is run by first  splitting the data set into a number of k-parts. Out of the k-parts, it will use 1 fold for testing and k minus one folds for training. It does this over a number of iterations (which is specified by cv=) and can output the mean of your desired metrics, like accuracy and precision.

In the below, C refers to regularization strength inverse. The smaller the value the stronger the regularization.
"""

# Grid search cross validation
#from sklearn.model_selection import GridSearchCV
#from sklearn.linear_model import LogisticRegression
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression(max_iter=1000)
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(x_train,y)

print("tuned hyperparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

"""According to the above, my best parameters are C: 10, which is the inverse of regularization strength. So lambda is 1/10, or .1

Penalty L2 (Ridge). Ridge will add “squared magnitude” of coefficient as a penalty term to the loss function

Accuracy given is .977

#### LogisticRegressionCV

I can also try LogisticRegressionCV to tune the hyperparameters
"""

#Try LogisticRegressionCV
from sklearn.linear_model import LogisticRegressionCV

clf = LogisticRegressionCV(
Cs=list(np.power(10.0, np.arange(-10, 10)))
#above checks all parameters between and 10
,penalty='l2'
,scoring='roc_auc'
,cv=5
,random_state=59
,max_iter=10000
,fit_intercept=True
,solver='newton-cg'
,tol=10)
searchCV.fit(x_train, y)

print ('Max auc_roc:', searchCV.scores_[1].max())

"""From the above, my max auc_roc accuracy is 0.97731, which is similar to the GridSearchCv.

I'll likely need to make some changes
"""

clf.fit(x_train, y)

print(clf.C_)
clf.scores_
#Below shows the output of the various AUC scores compiled

"""### Refit the model using new lambda


"""

#Import LogisticRegression and define regression run as LR2 (2 denotes penalty l2)

LR2 = LogisticRegression(penalty='l2', C=1.0, #or C=searchCV.C_[0], 
                           fit_intercept=True, 
                           solver='liblinear', max_iter=5000)
#Note solver liblinear is only to be used for one vs many schemes according to sklearn

"""## Run initial logistic regression

Below I will run an initial logistic regression without polynomials, undersampling or oversampling to have a baseline comparison
"""

y_hat_sc, y_hat_prob_sc, pred_test_sc, pred_test_prob_sc = binary_regression(LR2, x_train, y, test)

"""### Report which variable impacts more on results


"""

feature_impact(x_train, LR2)

"""### Evaluate initial model

The evaluate function will use the get_auc function and then ouput useful information, such as the confusion matrix, the classification report, the precision recall curve, and the roc curve. 
"""

#Evaluate initial model
#from sklearn.metrics import classification_report
evaluate('Imbalanced Scaled', LR2, x_train, y, y_hat_sc, y_hat_prob_sc)

"""Not horrible! But need to assess further

### Assess using cross-validation

As noted above, cross-validation, which is used as well to assess model accuracy, is run by first splitting the data set into a number of k-parts. Out of the k-parts, it will use 1 fold for testing and k minus one folds for training. It does this over a number of iterations (which is specified by cv=) and can output the mean of your desired metrics, like accuracy and precision.

For purpose of this notebook and best practices however, I will use a 10-fold CV to improve speed and efficiency as opposed to using too many folds
"""

y_pred_cv, y_prob_cv = binary_cvp(LR2, x_train, y, 10)

evaluate('Imbalanced CVP', LR2, x_train, y, y_pred_cv, y_prob_cv)

#Note accuracy is nearly 100%, which is misleading because of the unbalanced data.

"""Using cross validation, my predictions remained the same true negatives and true positives (incidence of Type 2 and Type 1 error fell slightly). My F1 score for predicting Krummholz forest type is still low (.63).

Because of the class-imbalance, my algorithm is very good at predicting when forests are not Krummholz as indicated by the high precision (indicating a low false positive rate, or Type 1 Error). This is easy though. The above recall is not very high, which indicates my false negative rate (Type 2 Error). 

Precision and recall are useful metrics when classes are very imbalanced. Thus, the algorithm is not good at predicting when forests are Krummholz (low precision). Sklearn indicates that "A system with high precision but low recall is just the opposite, returning very few results, but most of its predicted labels are correct when compared to the training labels."

Above, I have also plotted a precision recall curve, which plots precision on the y and recall on the x axis. In a precision recall curve, the higher the precision, the fewer number false positives. The higher the recall, the fewer number of false negatives. The curve looks ok, but can be improved to have a larger area

In other words, it's fairly simple for the algorithm to have high precision when there are so few Krummholz observations. We therefore most deal with class-imbalance.

# Undersampling and Oversampling

Undersampling and oversampling are both methods that can be used to make Cover_Type instances evenly distributed across all types. This is useful in some ways, but also can cause significant bias.

*   Oversampling can cause overfitting, because it duplicates the smaller classes so that there are enough observations to match the majority class

*   Undersampling results in information loss because it deletes observations from the majority class so that it coincides with minority class

Demonstrated visually below:

![image.png](data:image/png;base64,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)

## Undersampling
An undersampling technique will be used to cut the data down to equal number of observations for each cover_type. Although I lose a lot of data, the observations of Cover_Type will be equal across types.

### Random Undersampler from Imblearn

According to sklearn, this will under-sample the majority class by randomly picking samples with or without replacement.
"""

#Install imblearn
#pip install imbalanced-learn

#Create new data with y so I can make changes to a new dataset with representative number of y-variables
us_train = pd.concat([cover_types, x_train], axis=1)
rus_train, y_rus = separate_binary(us_train, 'Cover_Type', 7)

# Create an instance of RandomUnderSampler, fit the training data and apply Logistics regression
from imblearn.under_sampling import RandomUnderSampler

RUS = RandomUnderSampler()
rus_train, y_rus = RUS.fit_resample(rus_train, y_rus)

y_hat_rus, y_prob_rus, pred_test_rus, pred_prob_rus = binary_regression(LR, rus_train, y_rus, test)

evaluate('Random Undersampler', LR2, rus_train, y_rus, y_hat_rus, y_prob_rus)

#Assess accuracy using CVP
rus_pred_cv, rus_prob_cv = binary_cvp(LR2, rus_train, y_rus, 10)

evaluate('Random Undersampler CVP', LR, rus_train, y_rus, rus_pred_cv, rus_prob_cv)

"""It seems that my Random Undersampler improved my original scores and my curve, with F-1 Scores of .93 using CVP.

However, I'll try a few more undersampling techniques to see what's best.

### Cut down data and eliminate Cover_Type 4

Below I will try undersampling with a different more manual method I discovered. In this scenario, I will eliminate Cover_Type_4 as there are only 295 observations of it, and using an undersampling technique that drops me to 295 observations will results in too much data loss
"""

#From below I know that class 4 has the least number of observations with only 295.
cover_types.value_counts()

#Since I do not care about predicting class 4, I will  remove class 4 as it only has 295 observations
#and I do not want to cut every class down to 295
us_train = us_train[us_train.Cover_Type != 4]

#Get smallest class size in df
class_992 = np.min(us_train['Cover_Type'].value_counts().values)
print("Size of smallest class:", class_992)

#Cut down data to have Cover_Type 7 be evenly distributed

class_subsets = [us_train.query("Cover_Type == " + str(i)) for i in (list(range(1,4)) + list(range(5,8)))]

#This works by creating lists of subsets of data based on Cover_Type using the query function 
#and a for loop. An iterable is returned so it needs to be #converted to a list to function properly. 
#The + is used to skip over Cover_Type 4 which was removed

#Note cover_type 7 is the 6th item in my list
for i in range(6):
    class_subsets[i] = class_subsets[i].sample(class_992, replace=False, random_state=59)

us_train = pd.concat(class_subsets, axis=0).sample(frac=1.0, random_state=59).reset_index(drop=True)

#Double check that all Cover_types have 992 observations
us_train['Cover_Type'].value_counts()

"""### Run regression on undersampled data with 992 observations each"""

us_train, y_us = separate_binary(us_train, 'Cover_Type', 7)

y_pred_us, y_prob_us, pred_test_us, pred_prob_us = binary_regression(LR2, us_train, y_us, test)

evaluate('Undersampled Regression 995 Obs', LR, us_train, y_us, y_pred_us, y_prob_us)

"""In this scenario, my F1 score for predicting the non-Krummholz classes was increased to .98, however the True Positive predictor (of greater importance) was reduced to .88, as was my recall. It looks like the random undersampler was a better option. WIll check the CVP"""

us_pred_cv, us_prob_cv = binary_cvp(LR2, us_train, y_us, 10)

#y_hat_us_cv = cvp(LR, us_train, y_us, cv=50)
#y_prob_us_cv = cvp(LR, us_train, y_us, cv = 5, method = 'predict_proba')
evaluate('Undersampled CVP 995 Obs', LR2, us_train, y_us, us_pred_cv, us_prob_cv)

"""CVP confirms that technique was not as good as the random undersampler for my F1 score of the true positive.

###Eliminate Cover_Type 4 and 5

Perhaps eliminating Cover_Type_5 will improve things, as Cover_Type_5 is also rare  (only 1865 observations), and by dropping it I will lose less data. Use exact same technique.
"""

us_train2 = pd.concat([cover_types, x_train], axis=1)
#Drop cover types 4 and 5
us_train2 = us_train2[us_train2.Cover_Type != 4]
us_train2 = us_train2[us_train2.Cover_Type != 5]
#get min class size of 1865
class_1865 = np.min(us_train2['Cover_Type'].value_counts().values)

#Cut down data less to have Cover_Type 7 be evenly distributed

class_subsets = [us_train2.query("Cover_Type == " + str(i)) for i in (list(range(1,4)) + list(range(6,8)))]

#This works by creating lists of subsets of data based on Cover_Type using the query function 
#and a for loop. An iterable is returned so it needs to be #converted to a list to function properly. 
#The + is used to skip over Cover_Type 4 which was removed

#Note cover_type 7 is the 5th item in my list
for i in range(5):
    class_subsets[i] = class_subsets[i].sample(class_1865, replace=False, random_state=59)

#add new class subsets to df
us_train2 = pd.concat(class_subsets, axis=0).sample(frac=1.0, random_state=59).reset_index(drop=True)

#separate y
us_train2, y_us2 = separate_binary(us_train2, 'Cover_Type', 7)
#run regression
y_hat_us2, y_prob_us2, test_pred_us2, test_prob_us2 = binary_regression(LR2, us_train2, y_us2, test)
evaluate('Undersampled Regression 1865 obs.', LR2, us_train2, y_us2, y_hat_us2, y_prob_us2)

#Evaluate using CVP
y_hat_us_cv2, y_prob_us_cv2 = binary_cvp(LR2, us_train2, y_us2, 10)
evaluate('Undersampled Cross-Val-Pred 1865 Obs', LR2, us_train2, y_us2, y_hat_us_cv2, y_prob_us_cv2)

"""In the above using 1865 observations for each class and eliminating class types 4 and 5, I ended up with a lower AUC score and lower F1-score. Thus, I will stick with the random undersampler which I used in the first instance.

## Oversampling

Oversampling can also be attempted. In this case, I'll try the previous method and the SMOTE method

### Random oversampler
"""

#Reconcatenate and separate into binary
os_train = pd.concat([cover_types, x_train], axis=1)
ros_train, y_ros = separate_binary(os_train, 'Cover_Type', 7)

# Create an instance of RandomUnderSampler, fit the training data and apply Logistics regression
from imblearn.over_sampling import RandomOverSampler

ROS = RandomOverSampler()
ros_train, y_ros = ROS.fit_resample(ros_train, y_ros)

y_hat_ros, y_prob_ros, pred_test_ros, pred_prob_ros = binary_regression(LR2, ros_train, y_ros, test)

evaluate("Random Oversampler", LR2, ros_train, y_ros, y_hat_ros, y_prob_ros)

"""My AUC score went up slightly and I had F-1 Score of .92 for the true positive and negative"""

y_hat_ros_cv, y_prob_ros_cv = binary_cvp(LR2, ros_train, y_ros, 10)

evaluate("Random Oversampler CVP", LR2, ros_train, y_ros, y_hat_ros_cv, y_prob_ros_cv)

"""### SMOTE

I have more confidence in SMOTE than randomly oversampling because it creates synthetic data, so I don't have many identical data points with a standard oversampling method.

According to a Medium resource, the generation of synthetic data will work by first selecting random data from the minority class.
Then, the Euclidean distance between the random data and its k nearest neighbors will be calculated using k-nearest neighbor technique. Then, the difference will be multiplied with a random number between 0 and 1, then the result will be added to the minority class as a synthetic sample.

SMOTE will ideally reduce false negatives, but potentially increase recall (false positives) at the expense of precision (false negatives)
"""

from imblearn.over_sampling import SMOTE
# create smote datasets
smote_train = pd.concat([cover_types, x_train], axis=1)
smote_train, y_smote = separate_binary(os_train, 'Cover_Type', 7)

# Create an instance of SMOTE, fit the training data and apply #Logistics regression
smoter = SMOTE(random_state=59, sampling_strategy='minority', k_neighbors=5)
smote_train, y_smote = smoter.fit_resample(smote_train, y_smote)

smoter_classifier = LR2

smote_pred, smote_prob, smote_test, smote_test_prob = binary_regression(smoter_classifier, smote_train, y_smote, test)

evaluate('SMOTE', smoter_classifier, smote_train, y_smote, smote_pred,smote_prob)

"""Best scores yet! .93 F1 for the True Positive"""

kaggleattempt1_binary.to_csv("js_binary_1.csv", index=True, index_label='Index')

#Assess accuracy using CVP

y_hat_smote_cv, y_prob_smote_cv = binary_cvp(smoter_classifier, smote_train, y_smote, 10)

evaluate('SMOTE CVP', smoter_classifier, smote_train, y_hat_smote_cv, y_prob_smote_cv)

"""# Add Polynomial Features

In order to try and improve the above predictions, I will add polynomial features and run my best model on the new dataset
"""

#Add Polynomial Features (option to move this cell to before dummies have been re-added, but chose not to)
from sklearn.preprocessing import PolynomialFeatures
# perform a polynomial features transform of the dataset
poly = PolynomialFeatures(degree=2)
x_plf = poly.fit_transform(x_train)[:,1:]

test_plf = poly.fit_transform(test)[:,1:] # drop the intercept column
print(x_plf)
#intercept is not included

test_plf.shape

"""## Re-scale Data"""

#Must ensure to re-scale my data after adding many polynomial features
Scaler = StandardScaler()
Scaler.fit(x_plf)
x_plf = Scaler.transform(x_plf)
test_plf = Scaler.transform(test_plf)

"""## Tune hyperparameters

Pipeline was used below. According to sklearn:
"The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters. For this, it enables setting parameters of the various steps using their names and the parameter name separated by a '__', as in the example below. A step’s estimator may be replaced entirely by setting the parameter with its name to another estimator, or a transformer removed by setting it to 'passthrough' or None."
"""

#The below code initially ran for 7 hours and still did not get a result, so for the sake
#of submission I had to lower some parameters

from sklearn.linear_model import LogisticRegression

LR_poly = LogisticRegression(
    penalty='l2', 
    fit_intercept=True,
    max_iter = 500,
    )
x_train = x_plf.copy()
n_folds = 10 # remember that for this dataset this is leave-one-out
lambdas = np.array([0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
tuned_parameters = {'C': 1/lambdas} ## dictionary
clf = GridSearchCV(LR_poly, tuned_parameters, scoring = 'roc_auc', cv=n_folds, refit=False)
clf.fit(x_plf, y)

print(clf.best_score_)
print(clf.best_params_) 
#lambda = 1/C)
clf.cv_results_['mean_test_score']

"""## SMOTE with Polynomials"""

# create smote datasets
smote_train2 = x_plf
#y_smote already defined earlier
# Create an instance of SMOTE, fit the training data and apply #Logistics regression
smoter = SMOTE(random_state=59, sampling_strategy='minority', k_neighbors=5)
smote_train2, y_smote2 = smoter.fit_resample(x_plf, y)

smoter_classifier = LR2

smote_pred2, smote_prob2, smote_test2, smote_test_prob2 = binary_regression(smoter_classifier, smote_train2, y_smote2, test_plf)

evaluate("SMOTE Polynomial", smoter_classifier,smote_train2, y_smote2, smote_pred2, smote_prob2)

"""Looks like this did a great job with highest scores yet. Checking below with CVP (used 5-fold to save time given size of data set)"""

y_poly_smote_cv, y_probpoly_smote_cv = binary_cvp(smoter_classifier, smote_train2, y_smote2, 5)
evaluate("Smote CVP", smoter_classifier, smote_train2, _poly_smote_cv, y_probpoly_smote_cv)

submission(smote_test_prob2, 'Kaggle2_binary.csv')

"""Kaggle submission was .99135, so I feel comfortable moving to multiclass

### SMOTETomek

Attempting to improve upon last score with SMOTETomek, but don't have enough time
"""

st = SMOTETomek(smote=SMOTE(random_state=0), tomek=TomekLinks(sampling_strategy='majority'))
stx, y_st = st.fit_resample(x_plf, ym)

y_hat_st, y_prob_st, pred_test_st, pred_prob_st = binary_regression(LR2, stx, y_st, test_plf)
evaluate('SMOTETomek Binary', LR2, stx, y_st, y_hat_st)
