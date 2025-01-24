import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def display_skewness(data):
    skew_kurt_summary = {}
    for col in data.select_dtypes(include=['number']).columns:
        skew_kurt_summary[col] = {
            "Skewness": data[col].skew(),
            "Kurtosis": data[col].kurt()
        }
        print(f"{col}\nSkewness: {data[col].skew()}\nKurtosis: {data[col].kurt()}")
    return skew_kurt_summary

def find_missing_values(df):
    """
        Finds and summarizes missing values in the dataset.

        Parameters:
            df (pd.DataFrame): The dataset to analyze.

        Returns:
            pd.DataFrame: Summary table of missing values, percentages, and data types.
        """
    null_counts = df.isnull().sum()
    missing_value = null_counts
    percent_of_missing_value = 100 * null_counts / len(df)
    data_type = df.dtypes

    missing_data_summary = pd.concat([missing_value, percent_of_missing_value, data_type], axis=1)
    missing_data_summary_table = missing_data_summary.rename(columns={0: "Missing values", 1: "Percent of Total Values", 2: "DataType"})
    missing_data_summary_table = missing_data_summary_table[missing_data_summary_table.iloc[:, 1] != 0].sort_values('Percent of Total Values', ascending=False).round(1)

    print(f"From {df.shape[1]} columns selected, there are {missing_data_summary_table.shape[0]} columns with missing values.")

    return missing_data_summary_table

def boxPlotForDetectOutliers(data,column_names):
    """
       Plots box plots for numerical columns to detect outliers.

       Parameters:
           data (pd.DataFrame): The dataset containing numerical columns.
           column_names (list): List of column names to plot box plots.
       """
    for column in column_names:
        sns.boxplot(data=data[column])
        plt.title(f"Box Plot of {column}")
        plt.show()

def remove_outliers_winsorization(data,column_names):
    """
        Removes outliers from specified columns using winsorization (clipping).

        Parameters:
            data (pd.DataFrame): The dataset containing numerical columns.
            column_names (list): List of column names to apply winsorization.

        Returns:
            pd.DataFrame: Dataset with outliers clipped.
        """
    for column_name in column_names:
        q1 = data[column_name].quantile(0.25)
        q3 = data[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data[column_name] = data[column_name].clip(lower_bound, upper_bound)
    return data