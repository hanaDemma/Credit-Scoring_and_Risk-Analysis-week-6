
# Credit Scoring Model for Bati Bank

## Project Overview
This project aims to develop a **credit scoring model** for Bati Bank to enable a "buy-now-pay-later" system by assessing users' creditworthiness. The model classifies users as **high-risk** or **low-risk** based on their transaction data, helping the bank mitigate default risks and improve loan decision-making.

## Folder Structure 
CREDIT_SCORING_AND_RISK_ANALYSIS-WEEK6/
│
├── .github/
│
├── .week6/
│
├── .notebooks/
│   ├── exploratory_analysis.ipynb
│   └── README.md
│
├── .scripts/
│   ├── data_cleaning.py
│   ├── data_loader.py
│   └── feature_engineering.py
|   └── credit_risk_analysis.py
│
├── .src/
│
├── .tests/
│
├── .gitignore
│
├── app.py
│
├── .Dockerfile
│
├── README.md
│
└── requirements.txt


## Features

- **Data Preprocessing**: Handling missing values, feature engineering, and scaling.
- **Feature Engineering**: Creation of aggregate features and time-based features.
- **Encoding**: One-hot and label encoding of categorical variables.
- **Modeling**: Implementation of various machine learning algorithms to predict credit scores.


## Technologies

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

### Dataset Features
- `TransactionStartTime`: Timestamp of the transaction.
- `CustomerId`: Unique identifier for each customer.
- `Amount`: Transaction amount.
- `ProductCategory`: Category of the product purchased.
- `ChannelId`: Channel through which the transaction was made.
- `CurrencyCode`: Currency in which the transaction was made.
- And others

## Installation

To set up the project on your local machine, follow these steps:


1. Clone the repository:
   ```bash
   git clone https://github.com/hanaDemma/Credit-Scoring_and_Risk-Analysis-week-6.git
   cd Credit-Scoring_and_Risk-Analysis-week-6

2. pip install -r requirements.txt