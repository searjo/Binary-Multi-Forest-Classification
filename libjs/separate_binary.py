def separate_binary(train, column, target_value):
"""Separates the y from the training data and turns it into a binary variable
Parameters:
       train: training set
       column(str): your y column name to separate. Ensure it is in string format
    Returns:
       x and y dataframes
    """
    y = train[column]
    x = train.drop([column], axis=1)
    # x = x
    y = y == target_value
    y = y.astype(int)

    return x, y
