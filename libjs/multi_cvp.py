def multi_cvp(regression, x, y, folds):
""""""Runs cross_val_predict on your chosen data with chose number of k-folds
    Parameters:
           regression: your model which should be defined prior
           x: training data
           y: target variable
           folds: number of k-folds (10 is standard)
        Prints: Y Predictions and confusion matrix
        Returns:
        Predicted y for training
        """ 
 from sklearn.model_selection import cross_val_predict as cvp
  y_pred_cv = cvp(regression, x, y, cv=folds)
  #Turn back into df
  y_pred_cv = pd.DataFrame(y_pred_cv, columns = ['Y Prediction'])
  #Turn back into df and print output for prediction 
  print("TRAIN CVP PREDICTIONS", y_pred_cv)
  cmcvp =  confusion_matrix(y_pred=y_pred_cv, y_true=y)
  print ("Confusion Matrix", cmcvp)

  return y_pred_cv

