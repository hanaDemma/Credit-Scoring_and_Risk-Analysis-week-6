import pandas as pd
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