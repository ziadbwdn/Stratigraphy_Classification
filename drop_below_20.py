def drop_below_20 (X,target):
    '''
    function to remove dataset with target below 20 for effective train-test splitting
    param:
    X = dataset
    target = column target
    '''
    value_counts = X[target].value_counts()
    to_remove = value_counts[value_counts <= 20].index
    removed_data = X[X[target].isin(to_remove)]
    X = X[~X[target].isin(to_remove)]
    
    X_dropped = X.reset_index(drop=True)
    X_dropped[target].value_counts()
    
    print(f"Final total sample count: {len(X_dropped)}")
    return X_dropped