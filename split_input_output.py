# Create a function named split_input_output
def split_input_output(data,target_col):
    """
     Create a function named split_input_output - Has two arguments:
        - data, a pd Dataframe
        - target_col, a column (str)
        - Print the data shape after splitting
     then return X, y
    """
    # drop data
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    return X,y