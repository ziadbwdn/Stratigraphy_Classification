def EDA_Box_Plot(X, columns):
    # EDA Brief step

    y_target = 'Stratigraphy'
    num_columns = columns

    n_cols = 3
    n_rows = (len(num_columns) + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 4 * n_rows))

    for i, var in enumerate(num_columns):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.boxplot(x=y_target, y=var, data=X)
        plt.title(f'Box plot of {var} by {y_target}')
        plt.xlabel(y_target)
        plt.ylabel(var)

    plt.tight_layout()