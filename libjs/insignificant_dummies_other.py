def insignificant_dummies_other(dummy_col, threshold):
"""Turns insignificant dummies into other category
    Parameters:
           dummy_col: column to be changed
           threshold: threshold=int that you want use for dummy removal
         
        Returns:
        Dataframe with new dummies and other category
        """
    # removes the bind
    dummy_col = dummy_col.copy()

    # what is the ratio of a dummy in whole column
    count = pd.value_counts(dummy_col) / len(dummy_col)

    # cond whether the ratios are higher than the threshold
    mask = dummy_col.isin(count[count > threshold].index)

    # replace the ones which ratio is lower than the threshold by a special name
    dummy_col[~mask] = "Other"

    return pd.get_dummies(dummy_col, prefix=dummy_col.name)
