def multi_regression(regression, x, y, test):
	"""Runs a logistic regression on your chosen data
    Parameters:
           regression: your model which should be defined prior
           x: training data
           y: target variable
           test: testing data
        Prints: Important features and predictions for training and testing set
        Returns:
        Predicted y for training and testing data
        """
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
