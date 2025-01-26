import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

def calculateRFMSscores(new_dataframe_encoded, threshold=0):
    """
    Calculates RFMS (Recency, Frequency, Monetary, Seasonality) scores for customers and assigns classifications.

    Parameters:
        new_dataframe_encoded (pd.DataFrame): Dataset with encoded features.
        threshold (float): Threshold to classify customers as 'good' or 'bad' based on RFMS scores.

    Returns:
        pd.DataFrame: Dataset with calculated RFMS scores and classifications.
    """
    # Ensure TransactionStartTime is in datetime format
    new_dataframe_encoded['TransactionStartTime'] = pd.to_datetime(new_dataframe_encoded['TransactionStartTime'])

    # Set the current date (or use the last date in your dataset)
    current_date = new_dataframe_encoded['TransactionStartTime'].max()

    # Recency: Number of days since the last transaction
    recency_new_dataframe_encoded = new_dataframe_encoded.groupby('CustomerId').agg(
        {'TransactionStartTime': lambda x: (current_date - x.max()).days}
    )
    recency_new_dataframe_encoded.rename(columns={'TransactionStartTime': 'Recency'}, inplace=True)

    # Frequency: Count of transactions per customer
    frequency_new_dataframe_encoded = new_dataframe_encoded.groupby('CustomerId').agg({'TransactionId': 'count'})
    frequency_new_dataframe_encoded.rename(columns={'TransactionId': 'Frequency'}, inplace=True)

    # Monetary: Sum of transaction amounts per customer
    monetary_new_dataframe_encoded = new_dataframe_encoded.groupby('CustomerId').agg({'Amount': 'sum'})
    monetary_new_dataframe_encoded.rename(columns={'Amount': 'Monetary'}, inplace=True)

    # Seasonality: Average season (quarter) of transactions per customer
    new_dataframe_encoded['Season'] = new_dataframe_encoded['TransactionStartTime'].dt.quarter
    seasonality_new_dataframe_encoded = new_dataframe_encoded.groupby('CustomerId').agg({'Season': 'mean'})
    seasonality_new_dataframe_encoded.rename(columns={'Season': 'Seasonality'}, inplace=True)

    # Merge the dataframes into a single dataframe
    rfms_new_dataframe_encoded = recency_new_dataframe_encoded.merge(frequency_new_dataframe_encoded, on='CustomerId')
    rfms_new_dataframe_encoded = rfms_new_dataframe_encoded.merge(monetary_new_dataframe_encoded, on='CustomerId')
    rfms_new_dataframe_encoded = rfms_new_dataframe_encoded.merge(seasonality_new_dataframe_encoded, on='CustomerId')

    # Calculate RFMS Score
    rfms_new_dataframe_encoded['RFMS_Score'] = (
        rfms_new_dataframe_encoded['Recency'] * -1 +  # Lower recency is better
        rfms_new_dataframe_encoded['Frequency'] +
        rfms_new_dataframe_encoded['Monetary'] +
        rfms_new_dataframe_encoded['Seasonality']
    )

    # Classification: Assign 'good' or 'bad' based on the threshold
    rfms_new_dataframe_encoded['RiskCategory'] = rfms_new_dataframe_encoded['RFMS_Score'].apply(
        lambda score: 'good' if score >= threshold else 'bad'
    )

    return rfms_new_dataframe_encoded

def visualizeRFMSscore(rfms_new_dataframe_encoded):
    """
       Visualizes RFMS scores as a histogram.

       Parameters:
           rfms_new_dataframe_encoded (pd.DataFrame): Dataset containing RFMS scores.
       """
    # Plot the histogram of RFMS scores
    plt.figure(figsize=(8,6))
    plt.hist(rfms_new_dataframe_encoded['RFMS_Score'], bins=30, color='grey', alpha=0.7)
    plt.title('RFMS Score Distribution')
    plt.xlabel('RFMS Score')
    plt.ylabel('Frequency')
    plt.show()