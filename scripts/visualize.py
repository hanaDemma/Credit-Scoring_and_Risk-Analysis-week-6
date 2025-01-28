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


def visualize_categorical_features(dataframe):
    # Check for empty DataFrame
    if dataframe.empty:
        print("The DataFrame is empty.")
        return

    categorical_cols = dataframe.select_dtypes(include=['object']).columns

    if len(categorical_cols) == 0:
        print("No categorical columns found.")
        return

    sns.set(style='whitegrid')

    # Create a figure to hold the subplots for count plots
    plt.figure(figsize=(15, 4 * len(categorical_cols)))

    # Count plots for categorical features
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(len(categorical_cols), 1, i)
        sns.countplot(data=dataframe, x=col)
        plt.title(f'Distribution of {col}', fontsize=16)
        plt.xlabel(col, fontsize=14)
        plt.ylabel('Count', fontsize=14)

    plt.tight_layout()
    plt.show()



def correlation_analysis(dataframe):
    numerical_cols = dataframe.select_dtypes(include=['number']).columns
    dataframe = dataframe[numerical_cols[1:]]
    correlation_matrix = dataframe.corr()

    # Set plot style
    sns.set(style='whitegrid')

    # Create a heatmap to visualize the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()

def plot_confusion_matrix(cm):
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Random Forest')
    plt.show()


def plot_risk_scores(data):
    plt.hist(data['RiskScore'], bins=10, color='blue', edgecolor='black')
    plt.title('Distribution of Risk Scores')
    plt.xlabel('Risk Score')
    plt.ylabel('Frequency')
    plt.axvline(x=2, color='red', linestyle='--', label='Threshold = 2') 
    plt.legend()
    plt.show()


def plot_risk_counts(risk_counts):
    # Plot the classification results
    risk_counts.plot(kind='bar', color=['green', 'red'], alpha=0.7)
    plt.title('Risk Category Distribution')
    plt.xlabel('Risk Category')
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=0)
    plt.show()