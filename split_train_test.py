# Create a function named split_train_test

from sklearn.model_selection import train_test_split

def split_train_test(X, y, test_size, stratify, seed):
    '''Fungsi splitting data numerikal dan kategorikal
    memiliki dua argumen:
    - X, the input (pd.Dataframe)
    - y, the output (pd.Dataframe)
    - test_size, the test size between 0-1 (float)
    - seed, the random state (int)
    '''
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        stratify = y,
                                                        random_state = seed)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_train shape:", y_train.shape)

    return X_train, X_test, y_train, y_test