import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

def aggregate_features(data,by):
    aggregate_features = data.groupby(by).agg(
        Total_Transaction_Amount=('Amount', 'sum'),
        Average_Transaction_Amount=('Amount', 'mean'),
        Transaction_Count=('TransactionId', 'count'),
        Std_Deviation_Transaction_Amount=('Amount', 'std')
    ).reset_index()

    return aggregate_features


def time_correction(data):
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])

    # Extract features from the TransactionStartTime
    data['Transaction_Hour'] = data['TransactionStartTime'].dt.hour
    data['Transaction_Day'] = data['TransactionStartTime'].dt.day
    data['Transaction_Month'] = data['TransactionStartTime'].dt.month
    data['Transaction_Year'] = data['TransactionStartTime'].dt.year
    return data

def calculate_recency(df, customer_id_col, transaction_time_col):
    current_date = df[transaction_time_col].max()  # Latest date in the dataset
    recency = df.groupby(customer_id_col)[transaction_time_col].apply(lambda x: (current_date - x.max()).days)
    return recency

def calculate_frequency(df, customer_id_col, transaction_id_col):
    frequency = df.groupby(customer_id_col)[transaction_id_col].nunique()
    return frequency

def calculate_monetary(df, customer_id_col, transaction_amount_col):
    monetary = df.groupby(customer_id_col)[transaction_amount_col].sum()
    return monetary

def calculate_seasonality(df, customer_id_col, transaction_time_col):
    # Count the number of transactions per month for each customer
    df['Month'] = df[transaction_time_col].dt.to_period('M')  # Create a 'Month' column
    seasonality = df.groupby([customer_id_col, 'Month']).size().reset_index(name='Transactions')
    # Count the number of months with transactions to get a seasonality score
    seasonality_score = seasonality.groupby(customer_id_col)['Transactions'].count()
    return seasonality_score


def combine_rfms(df, customer_id_col):
    recency = calculate_recency(df, customer_id_col, 'TransactionStartTime')
    frequency = calculate_frequency(df, customer_id_col, 'TransactionId')
    monetary = calculate_monetary(df, customer_id_col, 'Amount')
    seasonality = calculate_seasonality(df, customer_id_col, 'TransactionStartTime')

    rfms_df = pd.DataFrame({
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': monetary,
        'Seasonality': seasonality
    }).fillna(0)  # Fill NaN values with 0 for customers with no transactions
    return rfms_df


def classify_customers_by_rfms(rfms_df):
    # Define thresholds using quantiles or other domain-specific rules
    rfms_df['RiskScore'] = (
        0.4 * pd.qcut(rfms_df['Recency'], 5, labels=False, duplicates='drop') +  # Recent transactions are better
        0.3 * pd.qcut(rfms_df['Frequency'], 5, labels=False, duplicates='drop') +  # Frequent transactions are better
        0.2 * pd.qcut(rfms_df['Monetary'], 5, labels=False, duplicates='drop') +   # Higher monetary value is better
        0.1 * pd.qcut(rfms_df['Seasonality'], 5, labels=False, duplicates='drop')  # More active seasons are better
    )

    # Classify based on RiskScore: High score = Good (low risk), Low score = Bad (high risk)
    rfms_df['RiskCategory'] = rfms_df['RiskScore'].apply(lambda x: 'Good' if x > 2.5 else 'Bad')
    return rfms_df

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
import pandas as pd
import numpy as np

def calculate_woe_iv(data, feature, target):
    """
       Calculates Weight of Evidence (WoE) and Information Value (IV) for a given feature.

       Parameters:
           data (pd.DataFrame): Dataset containing the feature and target variable.
           feature (str): The feature column to analyze.
           target (str): The target variable column.

       Returns:
           pd.DataFrame: DataFrame with unique values, WoE, IV, and total goods/bads.
           int, int: Total number of goods (target=1) and bads (target=0).
       """
    lst = []
    unique_values = data[feature].unique()
    total_good = len(data[data[target] == 1])
    total_bad = len(data[data[target] == 0])

    for val in unique_values:
        dist_good = len(data[(data[feature] == val) & (data[target] == 1)]) / total_good if total_good != 0 else 0
        dist_bad = len(data[(data[feature] == val) & (data[target] == 0)]) / total_bad if total_bad != 0 else 0

        # Handle cases where dist_good or dist_bad is zero
        if dist_good == 0:
            dist_good = 0.0001
        if dist_bad == 0:
            dist_bad = 0.0001

        woe = np.log(dist_good / dist_bad)
        iv = (dist_good - dist_bad) * woe
        lst.append({'Value': val, 'WoE': woe, 'IV': iv})

    # # Convert to DataFrame
    # woe_iv_df = pd.DataFrame(lst)

    # # Add totals to the output
    # woe_iv_df['Total_Good'] = total_good
    # woe_iv_df['Total_Bad'] = total_bad

    return pd.DataFrame(lst)


