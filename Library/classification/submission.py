


def submission(predictions, submission_filename):
  """Turns predictions into submittable format for Kaggle competition
      Parameters:
             predictions: test predictions for submission
             submission_filename: .csv filename
           
          Returns:
          CSV for submission
          """
  kaggleattempt = predictions.copy()
  kaggleattempt.index +=61006
  kaggleattempt.rename(columns={ kaggleattempt.columns[0]: 'Cover_Type' }, inplace = True)
  return kaggleattempt.to_csv(submission_filename, index=True, index_label='Index')
