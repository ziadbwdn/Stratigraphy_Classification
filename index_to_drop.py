def index_to_drop(X):
    '''
    function to drop X_train and y_train that have an anomalous tendency to our dataset
    parameter:
    1. X (using it to filter any index which have an anomaly required)
    output return: idx_to_drop list
    '''
    # Find the index of rows where data is nan
    X_isna_idx = X[X.isna().any(axis=1)].index.tolist()

    # Find the index of rows where data is null
    X_isnull_idx = X[X.isnull().any(axis=1)].index.tolist()

    # Find the index of rows where data is infinite
    X_infinite_idx = X[X.isin([np.inf, -np.inf]).any(axis=1)].index.tolist()

    # Find the index of rows where data containin negative values
    X_neg_idx = X[X.map(lambda x: isinstance(x, (int, float)) and x < 0).any(axis=1)].index.tolist()

    # convert it as as set then convert back to list
    drop_index = list(set([index for lst in [X_isna_idx, X_isnull_idx, X_infinite_idx, X_neg_idx] for index in lst]))
    idx_to_drop = []
    [idx_to_drop.append(i) for i in drop_index if i not in idx_to_drop]

    return idx_to_drop