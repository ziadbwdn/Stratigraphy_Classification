# Version Witout Scaling

from skbio.stats.composition import ilr, closure
from sklearn.preprocessing import OneHotEncoder


def preprocess_data(data, cat_imputer=None, cat_encoder=None, dataset_type = 'train'):
    """
    Preprocess data for training, validation, and test sets while avoiding data leakage.

    :param data: <pandas dataframe> input data
    :return: preprocessed data, percent_closure, percent_ilr
    """
    
    # Split data into numerical and categorical columns
    num_cols = data.select_dtypes(include=['number']).columns.tolist()
    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    num_data = data[num_cols]
    cat_data = data[cat_cols]

    if dataset_type == 'train':
         # Fit and transform the categorical data (train only)
        cat_imputer = SimpleImputer(strategy="constant", fill_value="UNKNOWN")
        cat_data_imputed = cat_imputer.fit_transform(cat_data)
        
        # Fit and transform the categorical encodings (train only)
        cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_data_encoded = cat_encoder.fit_transform(cat_data_imputed)
        # cat_encoded_data = pd.DataFrame(cat_data_encoded, columns=cat_cols)

    else: 
        # Categorical imputation (only transform)
        cat_data_imputed = cat_imputer.transform(cat_data)
        
        # Categorical encoding (only transform using the fitted encoders)
        cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        # Categorical imputation (transform only)
        cat_data_imputed = cat_imputer.transform(cat_data)

        # Categorical encoding using the fitted OneHotEncoder (transform only)
        cat_data_encoded = cat_encoder.transform(cat_data_imputed)
        # cat_encoded_data = pd.DataFrame(cat_data_encoded, columns=cat_cols)

    # Splitting numerical columns for ilr transformation
    percent_and_ppm_columns = ['Cr2O3_%', 'FeO_%', 'SiO2_%', 'Al2O3_%', 'CaO_%', 'MgO_%', 'P_%', 
                               'Au_ICP_ppm', 'Pt_ICP_ppm', 'Pd_ICP_ppm', 'Rh_ICP_ppm', 
                               'Ir_ICP_ppm', 'Ru_ICP_ppm']
    
    other_columns = ["MaxDepth", "DepthFrom", "DepthTo"]  # Non-ilr columns
    
    data_num_to_ilr = num_data[percent_and_ppm_columns].copy()
    data_num_non_ilr = num_data[other_columns].copy()
        
    # Convert ppm columns (not percentages) to same scale
    ppm_columns = [col for col in data_num_to_ilr.columns if col.endswith('_ppm')]
    data_num_to_ilr[ppm_columns] *= 0.0001  # Vectorized operation instead of apply
    
    # Get the percentage and ppm data as a NumPy array
    percent_data = data_num_to_ilr.to_numpy()

    # Normalization using closure
    percent_closure = closure(percent_data)

    # Avoid log(0) issues by replacing 0 with a small value
    epsilon = 1e-6
    percent_closure[percent_closure == 0] = epsilon

    # Perform ILR transformation
    percent_ilr = ilr(percent_closure)

    # ILR transformation returns one less dimension (11 columns from 12 inputs)
    ilr_transformed_columns = [f'ILR_{col}' for col in percent_and_ppm_columns[:-1]]
    num_data_ilr = pd.DataFrame(percent_ilr, columns=ilr_transformed_columns, index=data.index)

    # Concatenate numerical non-ilr and ilr-transformed columns
    num_concat = pd.concat([data_num_non_ilr, num_data_ilr], axis=1)

    # Concatenate numerical and one-hot encoded categorical data
    cat_data_encoded_df = pd.DataFrame(cat_data_encoded, index=data.index, columns=cat_encoder.get_feature_names_out(cat_cols))
    data_concat = pd.concat([cat_data_encoded_df, num_concat], axis=1)

    # Ensure all column names are strings
    data_concat.columns = data_concat.columns.astype(str)

    data_cleaned = data_concat

    # Check final output shape
    print(f"Expected shape: {data.shape}, Got: {data_cleaned.shape}")
    
    return data_cleaned, cat_imputer, cat_encoder, percent_closure, percent_ilr