import pandas as pd
import numpy as np
def dataLoading():
    """
       Loads a CSV file into a Pandas DataFrame.

       Returns:
           pd.DataFrame: DataFrame containing the loaded data.
       """
    return pd.read_csv('docs/data.csv')