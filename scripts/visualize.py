import matplotlib.pyplot as plt
import seaborn as sns


def visualize_numerical_features(dataframe):
    numerical_cols = dataframe.select_dtypes(include=['number']).columns

    # Set plot style
    sns.set(style='whitegrid')

    # Create a figure to hold the subplots for histogram and KDE
    plt.figure(figsize=(15, 4 * len(numerical_cols)))

    # Histograms with KDE
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(len(numerical_cols), 1, i)
        sns.histplot(dataframe[col], bins=10, kde=True)
        plt.title(f'Distribution of {col}', fontsize=16)
        plt.xlabel(col, fontsize=14)
        plt.ylabel('Frequency', fontsize=14)

    plt.tight_layout()
    plt.show()

    