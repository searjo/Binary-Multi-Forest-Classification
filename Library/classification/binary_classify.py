
def separate_binary(train, column, target_value):
"""Separates the y from the training data and turns it into a binary variable
Parameters:
       train: training set
       column(str): your y column name to separate. Ensure it is in string format
    Returns:
       x and y dataframes
    """
    y = train[column]
    x = train.drop([column], axis=1)
    # x = x
    y = y == target_value
    y = y.astype(int)

    return x, y


def binary_regression(regression, x, y):
"""Runs a logistic regression on your chosen data
Parameters:
       regression: your model which should be defined prior
       x: training data
       y: target variable
    Prints: Important features and predictions for training and testing set

    Returns:
    Predicted y and probability of y for training and testing data
    """
    regression.fit(x, y)
    print("Intercept is:", regression.intercept_)
    y_hat = regression.predict(x)
    # Turn back into df
    y_hat = pd.DataFrame(y_hat, columns=['Y Prediction'])
    # Get probability, one for both 0 and 1, and one for 1
    # Use [:,1] above to only return probability of one
    y_hat_prob = regression.predict_proba(x)
    y_hat_prob1 = regression.predict_proba(x)[:, 1]

    # Turn back into df and print output for prediction and probability
    y_hat_prob1 = pd.DataFrame(y_hat_prob1, columns=['Predicted Prob of Y=1'])
    output = pd.concat([y_hat, y_hat_prob1], axis=1)
    print("TRAIN PREDICTIONS")
    print(output)

    # Do same for testing set predictions
    print("TEST PREDICTIONS")
    pred_test = regression.predict(test)
    pred_test = pd.DataFrame(pred_test, columns=['Y Prediction Test'])
    pred_test_prob = regression.predict_proba(test)[:, 1]
    pred_test_prob = pd.DataFrame(pred_test_prob, columns=['Predicted Prob of Y=1 (Test)'])
    output_test = pd.concat([pred_test, pred_test_prob], axis=1)
    print(output_test)

    # Report which variable impacts more on results
    importance = regression.coef_[0]
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()

    return y_hat, y_hat_prob, pred_test, pred_test_prob

def scale_removetwodummies(train, test, dummycolumn1, dummycolumn2):
    """Scales the data and removes two dummies to prevent them from being scaled
    Parameters:
           train: training set
           test: testing set
           str(dummycolumn1): dummy variable you want removed
           str(dummycolumn2): dummy variable you want removed
        Prints: Important features and predictions for training and testing set

        Returns:
        Predicted y and probability of y for training and testing data
        """
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

def run_cvp(regression, x, y, folds):
    """Runs cross_val_predict on your chosen data with chose number of k-folds
    Parameters:
           regression: your model which should be defined prior
           x: training data
           y: target variable
           folds: number of k-folds (10 is standard)
        Prints: Y Predictions and confusion matrix

        Returns:
        Predicted y and probability of y for training and testing data
        """
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