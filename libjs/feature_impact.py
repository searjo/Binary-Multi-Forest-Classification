def feature_impact(x, regression):
 """Provides insight on the coefficients and impact of certain features
           x: training set
           regression: model used for fitting regression
        Returns:
        Bar plot of feature impact as well as table format for specific values
        """
  feature_impact = pd.DataFrame({"Feature":x.columns.tolist(),"Coefficients":regression.coef_[0]})
  print(feature_impact.plot.bar(x='Feature', y='Coefficients', color='green'))
  return feature_impact
