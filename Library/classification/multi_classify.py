def multi_regression(regression, x, y):
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
    #Turn back into df
    y_hat = pd.DataFrame(y_hat, columns = ['Y Prediction'])
    #Get probabilities, one for both 0 and 1, and one for 1
    #Use [:,1] above to only return probability of one
    y_hat_prob = regression.predict_proba(x)

    #Turn back into df and print output for prediction and probability
    y_hat_prob = pd.DataFrame(y_hat_prob, columns = ['Predicted Prob of Y True Pos'])
    output = pd.concat([y_hat, y_hat_prob], axis=1)
    print("TRAIN PREDICTIONS")
    print(output)

    #Do same for testing set predictions
    print("TEST PREDICTIONS")
    pred_test = regression.predict(test)
    pred_test = pd.DataFrame(pred_test, columns = ['Y Prediction Test'])
    pred_test_prob = regression.predict_proba(test)
    pred_test_prob = pd.DataFrame(pred_test_prob, columns = ['Predicted Prob of Y True Pos (Test)'])
    output_test = pd.concat([pred_test, pred_test_prob], axis=1)
    print(output_test)
    return y_hat, y_hat_prob, pred_test, pred_test_prob

def multi_cvp(regression, x, y, folds):
  from sklearn.model_selection import cross_val_predict as cvp
  y_pred_cv = cvp(regression, x, y, cv=folds)
  y_prob_cv = cvp(regression, x, y, cv=folds, method='predict_proba')
  #Turn back into df
  y_pred_cv = pd.DataFrame(y_pred_cv, columns = ['Y Prediction'])
  #Get probabilities
  y_prob_cv = regression.predict_proba(x)
  #Turn back into df and print output for prediction and probability
  y_prob_cv = pd.DataFrame(y_prob_cv, columns = ['Predicted Prob of Y True Pos CVP'])
  output = pd.concat([y_pred_cv, y_prob1_cv], axis=1)
  print("TRAIN CVP PREDICTIONS")
  print(output)
  cmcvp =  confusion_matrix(y_pred=y_pred_cv, y_true=y)
  print ("Confusion Matrix", cmcvp)