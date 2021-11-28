#Get new y to use
ym = cover_types.copy()

#Baseline regression. must use multi_class= in this instance for the regression to work. 
MLR = LogisticRegression(multi_class='multinomial', solver='lbfgs')

def separate_multi(train, column):
  y = train[column]
  x = train.drop([column], axis=1)
  return x, y

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
