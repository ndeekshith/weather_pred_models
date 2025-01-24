import warnings
warnings.filterwarnings("ignore")

# Import necessary modules
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, jaccard_score, f1_score, log_loss,
    mean_absolute_error, mean_squared_error, r2_score
)
import joblib

# --- Data Loading and Preprocessing ---

# Read the data
filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv"
df = pd.read_csv(filepath)

# Data preprocessing
# Convert 'RainToday' and 'RainTomorrow' to binary (0, 1)
df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1})
df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})

# One-hot encode categorical columns
df_processed = pd.get_dummies(data=df, columns=['WindGustDir', 'WindDir9am', 'WindDir3pm'])

# Drop the 'Date' column as it's not useful for modeling
df_processed.drop('Date', axis=1, inplace=True)

# Ensure all columns are of type float
df_processed = df_processed.astype(float)

# Define features (X) and target (y)
X = df_processed.drop('RainTomorrow', axis=1)
y = df_processed['RainTomorrow']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training and Evaluation ---

def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate a classification model and return metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Jaccard Index': jaccard_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'Log Loss': log_loss(y_test, y_pred_proba) if y_pred_proba is not None else np.nan
    }
    return metrics

# --- Linear Regression ---
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Calculate evaluation metrics for Linear Regression
LinearRegression_MAE = mean_absolute_error(y_test, y_pred_lr)
LinearRegression_MSE = mean_squared_error(y_test, y_pred_lr)
LinearRegression_R2 = r2_score(y_test, y_pred_lr)

print("Linear Regression Metrics:")
print(f"MAE: {LinearRegression_MAE:.4f}")
print(f"MSE: {LinearRegression_MSE:.4f}")
print(f"R2: {LinearRegression_R2:.4f}")

# --- KNN ---
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
KNN_metrics = evaluate_classification_model(knn, X_test, y_test)

print("\nKNN Metrics:")
for metric, value in KNN_metrics.items():
    print(f"{metric}: {value:.4f}")

# --- Decision Tree ---
tree = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
tree.fit(X_train, y_train)
Tree_metrics = evaluate_classification_model(tree, X_test, y_test)

print("\nDecision Tree Metrics:")
for metric, value in Tree_metrics.items():
    print(f"{metric}: {value:.4f}")

# --- Logistic Regression ---
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X, y, test_size=0.2, random_state=1)
lr_classifier = LogisticRegression(solver='liblinear', random_state=1)
lr_classifier.fit(X_train_lr, y_train_lr)
LR_metrics = evaluate_classification_model(lr_classifier, X_test_lr, y_test_lr)

print("\nLogistic Regression Metrics:")
for metric, value in LR_metrics.items():
    print(f"{metric}: {value:.4f}")

# --- SVM ---
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train, y_train)
SVM_metrics = evaluate_classification_model(svm, X_test, y_test)

print("\nSVM Metrics:")
for metric, value in SVM_metrics.items():
    print(f"{metric}: {value:.4f}")

# --- Model Comparison and Selection ---

# Create a DataFrame to compare model performance
metrics = {
    'Model': ['Linear Regression', 'KNN', 'Decision Tree', 'Logistic Regression', 'SVM'],
    'Accuracy': [np.nan, KNN_metrics['Accuracy'], Tree_metrics['Accuracy'], LR_metrics['Accuracy'], SVM_metrics['Accuracy']],
    'Jaccard Index': [np.nan, KNN_metrics['Jaccard Index'], Tree_metrics['Jaccard Index'], LR_metrics['Jaccard Index'], SVM_metrics['Jaccard Index']],
    'F1 Score': [np.nan, KNN_metrics['F1 Score'], Tree_metrics['F1 Score'], LR_metrics['F1 Score'], SVM_metrics['F1 Score']],
    'Log Loss': [np.nan, KNN_metrics['Log Loss'], Tree_metrics['Log Loss'], LR_metrics['Log Loss'], SVM_metrics['Log Loss']]
}
metrics_df = pd.DataFrame(metrics)
print("\nModel Comparison:")
print(metrics_df)

# --- Best Model Selection ---

# Select Logistic Regression as the best model
best_model = lr_classifier
joblib.dump(best_model, 'logistic_regression_model.pkl')
print("\nBest Model: Logistic Regression saved as 'logistic_regression_model.pkl'")

from sklearn.preprocessing import OneHotEncoder
import joblib

# Assuming you have a list of categorical features
categorical_features = ['WindDir9am', 'WindDir3pm', 'WindGustDir']

# Initialize the encoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Fit the encoder on your categorical data (example with a DataFrame `train_data`)
encoder.fit(df[categorical_features])

# Save the encoder
joblib.dump(encoder, 'encoder.pkl')
