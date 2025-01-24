import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

def aggregateFeatures(data):
    """
       Aggregates transaction data by customer.

       Parameters:
           data (pd.DataFrame): The dataset containing transaction details.

       Returns:
           pd.DataFrame: Aggregated data with total, mean, count, and std of transaction amounts.
       """
    # Aggregate features by customer (AccountId)
    agg_data = data.groupby('AccountId').agg(
        TotalTransactionAmount=('Amount', 'sum'),
        AverageTransactionAmount=('Amount', 'mean'),
        TransactionCount=('TransactionId', 'count'),
        StdTransactionAmount=('Amount', 'std')
    ).reset_index()
    return agg_data

def extractDateAndTime(new_dataframe):
    """
        Extracts date and time features from a datetime column.

        Parameters:
            new_dataframe (pd.DataFrame): Dataset containing a datetime column 'TransactionStartTime'.

        Returns:
            pd.DataFrame: Dataset with new time-based features.
        """
    # Converting TransactionStartTime to datetime format
    new_dataframe['TransactionStartTime'] = pd.to_datetime(new_dataframe['TransactionStartTime'])

    # Extracting date and time-based features
    new_dataframe['TransactionHour'] = new_dataframe['TransactionStartTime'].dt.hour
    new_dataframe['TransactionDay'] = new_dataframe['TransactionStartTime'].dt.day
    new_dataframe['TransactionMonth'] = new_dataframe['TransactionStartTime'].dt.month
    new_dataframe['TransactionYear'] = new_dataframe['TransactionStartTime'].dt.year
    return new_dataframe

def encodingCategoricalVariables(new_dataframe):
    """
        Encodes categorical variables using OneHotEncoding.

        Parameters:
            new_dataframe (pd.DataFrame): Dataset containing categorical columns.

        Returns:
            pd.DataFrame: Dataset with encoded categorical variables.
        """
    categorical_columns = ['CurrencyCode', 'ProductCategory']
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_data = encoder.fit_transform(new_dataframe[categorical_columns])
    encoded_new_dataframe = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
    new_dataframe_encoded = pd.concat([new_dataframe.reset_index(drop=True), encoded_new_dataframe], axis=1)
    new_dataframe_encoded.drop(columns=categorical_columns, inplace=True)
    return new_dataframe_encoded