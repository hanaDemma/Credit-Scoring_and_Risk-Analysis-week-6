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