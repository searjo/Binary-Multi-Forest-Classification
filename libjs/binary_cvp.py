def binary_cvp(regression, x, y, folds):
 """Runs cross_val_predict on your chosen data with chose number of k-folds
    Parameters:
           regression: your model which should be defined prior
           x: training data
           y: target variable
           folds: number of k-folds (10 is standard)
        Prints: Y Predictions and confusion matrix
        Returns:
        Predicted y and probability of y for training  data
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
