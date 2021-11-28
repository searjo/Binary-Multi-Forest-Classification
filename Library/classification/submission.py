


def submission(predictions, submission_name):
  """Turns predictions into submittable format for Kaggle competition
      Parameters:
             p: training set
             test: testing set
             str(dummycolumn1): dummy variable you want removed
             str(dummycolumn2): dummy variable you want removed
          Prints: Important features and predictions for training and testing set

          Returns:
          Predicted y and probability of y for training and testing data
          """
  kaggleattempt = predictions.copy()
  kaggleattempt.index +=61006
  kaggleattempt.rename(columns={ kaggleattempt.columns[1]: 'Cover_Type' }, inplace = True)
  return kaggleattempt.to_csv(submission_name, index=True, index_label='Index')