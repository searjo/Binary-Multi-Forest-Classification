def separate_multi(train, column):
"""Separates y from x
	Parameters: 
	train: training data
	str(column): column to be removed (string)
  y = train[column]
  x = train.drop([column], axis=1)
  return x, y

