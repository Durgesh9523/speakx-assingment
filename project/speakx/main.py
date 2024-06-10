import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Convert 'TotalCharges' to numeric, coerce errors to NaN
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Handle missing values in 'TotalCharges'
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

# Encode binary categorical variables
data['gender'] = data['gender'].map({'Female': 1, 'Male': 0})
data['Partner'] = data['Partner'].map({'Yes': 1, 'No': 0})
data['Dependents'] = data['Dependents'].map({'Yes': 1, 'No': 0})
data['PhoneService'] = data['PhoneService'].map({'Yes': 1, 'No': 0})
data['PaperlessBilling'] = data['PaperlessBilling'].map({'Yes': 1, 'No': 0})
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# Encode other categorical variables using one-hot encoding
data = pd.get_dummies(data, columns=[
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod'
])
scaler = StandardScaler()
data[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(data[['tenure', 'MonthlyCharges', 'TotalCharges']])
sns.countplot(x='Churn', data=data)
plt.show()

# Boxplot of tenure vs churn
sns.boxplot(x='Churn', y='tenure', data=data)
plt.show()

#Creating a feature for total services subscribed
data['TotalServices'] = (data[['PhoneService', 'MultipleLines_Yes', 'InternetService_DSL',
                               'InternetService_Fiber optic', 'OnlineSecurity_Yes',
                               'OnlineBackup_Yes', 'DeviceProtection_Yes', 'TechSupport_Yes',
                               'StreamingTV_Yes', 'StreamingMovies_Yes']].sum(axis=1))


# Split the data
X = data.drop(['customerID', 'Churn'], axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluation
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

lr_metrics = evaluate_model(y_test, y_pred_lr)
rf_metrics = evaluate_model(y_test, y_pred_rf)

print(f'Logistic Regression: Accuracy={lr_metrics[0]}, Precision={lr_metrics[1]}, Recall={lr_metrics[2]}, F1-Score={lr_metrics[3]}')
print(f'Random Forest: Accuracy={rf_metrics[0]}, Precision={rf_metrics[1]}, Recall={rf_metrics[2]}, F1-Score={rf_metrics[3]}')