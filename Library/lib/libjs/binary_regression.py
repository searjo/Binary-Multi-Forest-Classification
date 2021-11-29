def binary_regression(regression, x, y, test):
"""Runs a logistic regression on your chosen data
Parameters:
       regression: your model which should be defined prior
       x: training data
       y: target variable
       test: test data
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

    return y_hat, y_hat_prob, pred_test, pred_test_prob
