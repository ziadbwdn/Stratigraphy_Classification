def read_data(fname):
    """
    Fungsi untuk load data & hapus duplikat
    :param filename: <string> nama file input (format .csv)
    :return data: <pandas dataframe> sampel data
    """

    # read data
    data = pd.read_csv(fname, sep = ';')
    print("Data asli            : ", data.shape, "- (#observasi, #kolom)")
    
    # checking value missing, infinite, negative, and duplicated
    data_duplicated = data.duplicated().sum()

    # report from the loaded dataset
    print("duplicated data:", data_duplicated, " -(#observasi duplikasi, #nomor)")
    
    # final checking dataset after the duplicate value get dropped
    df = pd.DataFrame(data)
    for col in data.columns:
        if 'Filter' in col:
            del data[col]
    data.drop_duplicates()
    print("Data shape final      : ", data.shape, "- (#observasi, #kolom)")

    return data