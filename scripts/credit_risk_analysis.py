from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
from matplotlib import cm
import matplotlib.pyplot as plt



def prepare_for_model(data):
    X = data.drop(
        ['Month', 'TransactionStartTime', 'TransactionId', 'AccountId', 'CustomerId', 'RiskCategory', 'BatchId',
         'SubscriptionId'], axis=1)  # Exclude identifiers and target
    X = data[['TotalRFMS', 'ProviderId_ProviderId_2', 'ProviderId_ProviderId_3',
              'ProviderId_ProviderId_4', 'ProviderId_ProviderId_5', 'ProviderId_ProviderId_6',
              'ProductId_ProductId_10', 'ProductId_ProductId_11', 'ProductId_ProductId_12',
              'ProductId_ProductId_13', 'ProductId_ProductId_14', 'ProductId_ProductId_15',
              'ProductId_ProductId_16', 'ProductId_ProductId_19', 'ProductId_ProductId_2',
              'ProductId_ProductId_20', 'ProductId_ProductId_21', 'ProductId_ProductId_22',
              'ProductId_ProductId_23', 'ProductId_ProductId_24', 'ProductId_ProductId_27',
              'ProductId_ProductId_3', 'ProductId_ProductId_4', 'ProductId_ProductId_5',
              'ProductId_ProductId_6', 'ProductId_ProductId_7', 'ProductId_ProductId_8',
              'ProductId_ProductId_9', 'ProductCategory_data_bundles', 'ProductCategory_financial_services',
              'ProductCategory_movies', 'ProductCategory_other', 'ProductCategory_ticket',
              'ProductCategory_transport', 'ProductCategory_tv', 'ProductCategory_utility_bill',
              'ChannelId_ChannelId_2', 'ChannelId_ChannelId_3', 'ChannelId_ChannelId_5',
              'Amount', 'Value', 'PricingStrategy', 'FraudResult', 'Total_Transaction_Amount',
              'Average_Transaction_Amount', 'Transaction_Count', 'Std_Deviation_Transaction_Amount', 'Transaction_Hour',
              'Transaction_Day', 'Transaction_Month', 'Transaction_Year']]
    
    X = data[['TotalRFMS', 'ProviderId_ProviderId_2', 'ProviderId_ProviderId_3',
              'ProviderId_ProviderId_4', 'ProviderId_ProviderId_5', 'ProviderId_ProviderId_6',
              'ProductId_ProductId_10', 'ProductId_ProductId_11', 'ProductId_ProductId_12',
              'ProductId_ProductId_13', 'ProductId_ProductId_14', 'ProductId_ProductId_15',
              'ProductId_ProductId_16', 'ProductId_ProductId_19', 'ProductId_ProductId_2',
              'ProductId_ProductId_20', 'ProductId_ProductId_21', 'ProductId_ProductId_22',
              'ProductId_ProductId_23', 'ProductId_ProductId_24', 'ProductId_ProductId_27',
              'ProductId_ProductId_3', 'ProductId_ProductId_4', 'ProductId_ProductId_5',
              'ProductId_ProductId_6', 'ProductId_ProductId_7', 'ProductId_ProductId_8',
              'ProductId_ProductId_9', 'ProductCategory_data_bundles', 'ProductCategory_financial_services',
              'ProductCategory_movies', 'ProductCategory_other', 'ProductCategory_ticket',
              'ProductCategory_transport', 'ProductCategory_tv', 'ProductCategory_utility_bill',
              'ChannelId_ChannelId_2', 'ChannelId_ChannelId_3', 'ChannelId_ChannelId_5',
              'Amount', 'Value', 'PricingStrategy', 'FraudResult', 'Total_Transaction_Amount',
              'Average_Transaction_Amount', 'Transaction_Count', 'Transaction_Hour',
              'Transaction_Day', 'Transaction_Month', 'Transaction_Year']]
    y = data['RiskCategory'].map({'Good': 0, 'Bad': 1})  # Binary mapping for classification


    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train.to_csv('docs/X_train_data.csv')
    y_train.to_csv('docs/y_train_data.csv')
    X_test.to_csv('docs/X_test_data.csv')
    y_test.to_csv('docs/y_test_data.csv')
    return X_train, X_test, y_train, y_test, X, y


def modeling(X_train, X_test, y_train, y_test,results):
    # 1. Logistic Regression
    log_model = LogisticRegression(max_iter=1000, C=0.1, penalty='l2')
    log_model.fit(X_train, y_train)
    log_preds = log_model.predict(X_test)
    log_probs = log_model.predict_proba(X_test)[:, 1]

    results['Logistic Regression'] = {
        'Accuracy': accuracy_score(y_test, log_preds),
        'ROC AUC': roc_auc_score(y_test, log_probs),
        'Classification Report': classification_report(y_test, log_preds,zero_division=1)
    }
    # Save the model
    joblib.dump(log_model, 'model/logistic_regression_model.pkl')

    # 2. Random Forest
    rf_model = RandomForestClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]

    results['Random Forest'] = {
        'Accuracy': accuracy_score(y_test, rf_preds),
        'ROC AUC': roc_auc_score(y_test, rf_probs),
        'Classification Report': classification_report(y_test, rf_preds,zero_division=1)
    }
    # Save the model
    joblib.dump(rf_model, 'model/random_forest_model.pkl')

    # 3. Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=3, min_samples_split=10, min_samples_leaf=5, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    dt_probs = dt_model.predict_proba(X_test)[:, 1]

    results['Decision Tree'] = {
        'Accuracy': accuracy_score(y_test, dt_preds),
        'ROC AUC': roc_auc_score(y_test, dt_probs),
        'Classification Report': classification_report(y_test, dt_preds, zero_division=1)
    }
    # Save the model
    joblib.dump(dt_model, 'model/decision_tree_model.pkl')

    # 4. Gradient Boosting Machine
    gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, min_samples_split=10,
                                          random_state=42)
    gb_model.fit(X_train, y_train)
    gb_preds = gb_model.predict(X_test)
    gb_probs = gb_model.predict_proba(X_test)[:, 1]

    results['Gradient Boosting'] = {
        'Accuracy': accuracy_score(y_test, gb_preds),
        'ROC AUC': roc_auc_score(y_test, gb_probs),
        'Classification Report': classification_report(y_test, gb_preds, zero_division=1)
    }

    # Save the model
    joblib.dump(gb_model, 'model/gradient_boosting_model.pkl')

    return results, log_preds, dt_preds, gb_preds, rf_preds


# Function for hyperparameter tuning using RandomizedSearchCV
def tune_random_forest(X_train, y_train):
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2']
    }
    
    rf_search = RandomizedSearchCV(
        rf, param_distributions=param_grid, 
        n_iter=20,  # Fewer iterations for speed
        cv=5, 
        scoring='roc_auc', 
        n_jobs=-1, 
        random_state=42
    )
    rf_search.fit(X_train, y_train)
    print("Best Hyperparameters:", rf_search.best_params_)
    print(f"Best ROC-AUC: {rf_search.best_score_:.3f}")
    
    return rf_search  # Return the RandomizedSearchCV object

# Function to plot feature importances for Random Forest
def plot_feature_importance(model, X_train):
    # Extract feature importances and features
    feature_importance = model.feature_importances_
    features = X_train.columns

    # Sort feature importances in descending order
    sorted_idx = feature_importance.argsort()

    # Generate a colormap
    cmap = cm.get_cmap('viridis', len(features))  # Use any colormap like 'viridis', 'plasma', etc.
    colors = [cmap(i) for i in range(len(features))]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(features[sorted_idx], feature_importance[sorted_idx], color=np.array(colors)[sorted_idx])
    plt.xlabel("Feature Importance", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.title("Feature Importance (Random Forest)", fontsize=14, fontweight="bold")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()



def save_best_model(results,X_train, y_train, X_test, y_test):
    best_params = {
        'subsample': 0.9,
        'n_estimators': 500,
        'max_depth': 7,
        'learning_rate': 0.1
    }

    # Initialize the Gradient Boosting model with the best parameters
    gb_model = GradientBoostingClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        random_state=42
    )

    gb_model.fit(X_train, y_train)
    gb_preds = gb_model.predict(X_test)
    gb_probs = gb_model.predict_proba(X_test)[:, 1]

    results['Gradient Boosting'] = {
        'Accuracy': accuracy_score(y_test, gb_preds),
        'ROC AUC': roc_auc_score(y_test, gb_probs),
        'Classification Report': classification_report(y_test, gb_preds, zero_division=1)
    }

    # Save the model
    joblib.dump(gb_model, '../models/gradient_boosting_model.pkl')
    return gb_preds