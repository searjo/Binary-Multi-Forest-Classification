def scale_min2dum(train, test, dummycolumn1, dummycolumn2):
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
