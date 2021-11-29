def get_predictions(model, x, test):
 """Runs a logistic regression on your chosen data
    Parameters:
           regression: your model which should be defined prior
           x: training data
           test: testing data
        Prints: Important predictions for training and testing set
        Returns:
        Predicted y for training and testing data
        """
  y_pred = model.predict(x)
  y_pred_prob = model.predict_proba(x)
  y_pred_test = model.predict(test)
  print("Train Predictions:", y_pred)
  print("Test Predictions:", y_pred_test)
  y_pred = pd.DataFrame(y_pred, columns = ['Y Prediction'])
  y_pred_test = pd.DataFrame(y_pred_test, columns = ['Y Test Prediction'])
  return y_pred, y_pred_prob, y_pred_test
