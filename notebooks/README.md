###  Task 1 - Understanding Credit Risk
## Overview
Credit risk is the potential risk of loss due to a borrower’s failure to repay a loan or meet contractual obligations. It is a critical concept in financial institutions, especially for lenders, as they assess and manage the likelihood that they will not recover the money they lend.

This task aims to introduce key concepts and frameworks used to assess credit risk. By understanding credit risk, institutions can make informed decisions about lending and setting appropriate interest rates, limits, and policies.

# Task-2 Exploratory Data Analysis (EDA)

## Overview
Exploratory Data Analysis (EDA) is a critical step in the data science workflow, allowing us to understand the structure and characteristics of the dataset. In this task, we will perform various analyses to gain insights into the data, including distributions of features, correlations, missing values, and outlier detection.

## Steps:
- **Overview of the Data** : Understand the structure of the dataset.
- **Summary Statistics**: Understand central tendencies, dispersion, and shape of the dataset’s distribution.
- **Distribution of Numerical Features**:Visualize and understand the distribution of numerical features to identify patterns, skewness, and potential outliers.
- **Distribution of Categorical Features**: Analyze the distribution of categorical features to understand the frequency and variability of categories.
- **Correlation Analysis**: Understand relationships between numerical features.
- **Identifying Missing Values**: Detect missing values and determine appropriate strategies for handling them.
- **Outlier Detection**:Detect and handle outliers that may skew analysis or model performance


# Development Instructions
- Create a feature/task1-2 Branch for development.
- Commit progress regularly with clear and detailed commit messages.
- Merge updates into the main branch via a Pull Request (PR).

# Task 3: Feature Engineering

## Overview
Feature engineering is the process of transforming raw data into features that can improve the performance of machine learning models. In this task, we will focus on creating new features from the existing data, encoding categorical variables, handling missing values, and scaling numerical features to enhance model performance in credit scoring and risk analysis.

## Steps:

- **Create Aggregate Features**: Summarize transaction-level data to capture customer behavior.
- **Extract Features**: Extract additional features from existing data to capture time-based patterns.
- **Encode Categorical Variables**:Convert categorical variables into a format that machine learning models can understand.
- **Handle Missing Values**: Scale numerical features to ensure that all features contribute equally to the model

# Development Instructions
- Create a feature/task-3 Branch for development.
- Commit progress regularly with clear and detailed commit messages.
- Merge updates into the main branch via a Pull Request (PR).

# Task 4: Default estimator and WoE binning
## Overview
The purpose of a credit scoring system is to classify borrowers into risk categories, such as high-risk or low-risk, based on their likelihood of defaulting on a loan. High-risk individuals have a greater probability of failing to repay their loans on time. In this task, this task focus on creating a default estimator that predicts the probability of default and also Weight of Evidence (WoE) binning.
## Steps:
1. **Construct a default estimator (proxy)**
  - visualizing all transactions in the RFMS space, establish a boundary where users are classified as high and low RFMS scores.
  - Assign all users the good and bad label
2. **Perform Weight of Evidence (WoE) binning**

# Development Instructions
- Create a feature/task-4 Branch for development.
- Commit progress regularly with clear and detailed commit messages.
- Merge updates into the main branch via a Pull Request (PR).

# Task 5: Modelling
## Steps:
1. **Model Selection and Training**
      - Split the Data
      - Choose Models
      - Train the Models
      - Hyperparameter Tunning
2. **Model Evaluation**
    - Assess model performance using **Accuracy**, **Precision**,**Recall**,**F1 Score**, **ROC-AUC**  metrics
    
# Development Instructions
- Create a feature/task-5 Branch for development.
- Commit progress regularly with clear and detailed commit messages.
- Merge updates into the main branch via a Pull Request (PR).

# Task 6: Modelling Model Serving API Call

- Create a REST API to serve the trained machine-learning models for real-time predictions.
- **Choose a framework**:Select a suitable framework for building REST APIs 
- **Load the model**:Use the model from Task 4 to load the trained machine-learning model.
- **Define API endpoints**: Create API endpoints that accept input data and return predictions.
- **Handle requests**:Implement logic to receive input data, preprocess it, and make predictions using the loaded model.
- **Return predictions**: Format the predictions and return them as a response to the API call.
- **Deployment**: Deploy the API to a web server or cloud platform.

# Development Instructions
- Create a feature/task-6 Branch for development.
- Commit progress regularly with clear and detailed commit messages.
- Merge updates into the main branch via a Pull Request (PR).
