import pandas as pd
import numpy as np
import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def clean(df):

    df['nameDest'] = np.where(df['nameDest'].str.startswith('M'), 0,
                            np.where(df['nameDest'].str.startswith('C'), 1, -1)).astype(int)

    scaler = StandardScaler()
    df['amount_scaled'] = scaler.fit_transform(df[['amount']])

    df = df.drop(columns=['isFlaggedFraud', 'oldbalanceOrg', 'step', 'type', 'nameOrig',
                         'amount', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'], axis=1)

    df['isFraud'] = df.pop('isFraud')

    return df

def feature_engineering(df):
    df['emptied'] = np.nan
    for i in df.index[:-1]:  
        if df.at[i, 'type'] == "TRANSFER" and df.at[i+1, 'type'] == "CASH_OUT":
            if df.at[i, 'amount'] == df.at[i+1, 'amount']:
                df.at[i, 'emptied'] = int(1)
                df.at[i+1, 'emptied'] = int(1)
    df['emptied'] = df['emptied'].fillna(0)

    df['wealthy_customer'] = np.nan
    for i in df.index:  
        if df.loc[i, 'amount'] >= 1000000 and df.loc[i,'oldbalanceOrg'] >= 1000000:
            df.loc[i, 'wealthy_customer'] = int(1)
    df['wealthy_customer'] = df['wealthy_customer'].fillna(0)
    return df

def encoder(df):
    encoder = OneHotEncoder(sparse_output=False)  # Return a dense array
    encoded_data = encoder.fit_transform(df[['type']])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['type']))

    df = pd.concat([df, encoded_df], axis=1)
    return df

def custom_stratified(df):
    set_A = df[df['isFraud'] == 1]
    set_B = df[df['isFraud'] == 0]

    # Step 2: Construct the training set
    train_A = set_A.sample(frac=0.8, random_state=42)
    train_B = set_B.sample(n=len(train_A), random_state=42)  # n is the number of samples in train_A
    training_set = pd.concat([train_A, train_B])

    # Step 3: Construct the test set
    test_A = set_A.drop(train_A.index)
    ratio_B = len(set_B) / len(df)
    required_B_samples = int((len(test_A) / (1 - ratio_B)) * ratio_B) - len(test_A)
    test_B = set_B.drop(train_B.index).sample(n=required_B_samples, random_state=42)
    test_set = pd.concat([test_A, test_B])

    return training_set, test_set

def original_df(df):

    ## One Hot Encoder
    df = encoder(df)

    ## Feature engineering ##
    df = feature_engineering(df)

    ## Cleaning ##
    df = clean(df)

    ## partition ##
    training_set, test_set = custom_stratified(df)

    return training_set, test_set
