# Imports
import pandas as pd
import numpy as np
import pyodbc
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Connect to the Database and Fetch Data
# Database connection details
def fetch_data_from_database():
    server = 'DESKTOP-PNV6CGS'
    database = 'DEPI Final Project'
    connection_string = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'
    conn = pyodbc.connect(connection_string)

    # Query to fetch data
    query = """
    SELECT 
        t.transaction_date,
        t.quantity,
        p.product_cost,
        p.product_retail_price,
        c.customer_id,
        c.yearly_income,
        c.birthdate,
        c.gender,
        c.member_card
    FROM [DEPI Final Project].[dbo].[Transactions] t
    JOIN [DEPI Final Project].[dbo].[Product] p ON t.product_id = p.product_id
    JOIN [DEPI Final Project].[dbo].[Customer] c ON t.customer_id = c.customer_id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Step 3: Data Cleaning and Preprocessing
def preprocess_data(df):
    # Convert 'birthdate' to datetime, handling non-date values with errors='coerce'
    df['birthdate'] = pd.to_datetime(df['birthdate'], format='%m/%d/%Y', errors='coerce')

    # Drop rows with invalid dates
    df = df.dropna(subset=['birthdate'])

    # Replace birthdate with age
    df['age'] = pd.Timestamp('now').year - df['birthdate'].dt.year
    df.drop(columns=['transaction_date', 'birthdate', 'customer_id'], inplace=True)

    # Process yearly income column
    pattern = r"\$(\d+)K - \$(\d+)K"
    df['min_salary'] = df['yearly_income'].apply(
        lambda x: int(re.match(pattern, x).group(1)) * 1000 if re.match(pattern, x) else None)
    df['max_salary'] = df['yearly_income'].apply(
        lambda x: int(re.match(pattern, x).group(2)) * 1000 if re.match(pattern, x) else None)
    df['average_salary'] = (df['min_salary'] + df['max_salary']) // 2
    df.drop(columns=['yearly_income', 'min_salary', 'max_salary'], inplace=True)

    # Label Encoding
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['member_card'] = le.fit_transform(df['member_card'])

    # Ensure 'quantity' column is numeric for comparison
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)

    # Set threshold for quantity
    threshold = 4
    df['is_above_threshold'] = df['quantity'].apply(lambda x: 1 if x > threshold else 0)
    df.drop(columns=['quantity'], inplace=True)

    return df

# Step 4: Exploratory Data Analysis
def visualize_data(df):
    # Average Salary distribution
    sns.histplot(df['average_salary'])
    plt.title('Average Salary Distribution')
    plt.show()

    # Correlation matrix
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

# Step 5: Train and Evaluate the Model
def train_decision_tree_model(df):
    # Split data into features and target
    X = df[['age', 'gender', 'member_card', 'average_salary']]
    y = df['is_above_threshold']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a decision tree classifier
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)

    # Predict on the test set
    y_pred = classifier.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Classification Report: \n{classification_report(y_test, y_pred)}')

    # Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

    return classifier, scaler

# Step 6: Predict for a New Customer
def predict_new_customer(classifier, scaler, age, gender, member_card, salary):
    # Create a DataFrame with the same feature names for prediction
    input_data = pd.DataFrame([[age, gender, member_card, salary]],
                              columns=['age', 'gender', 'member_card', 'average_salary'])

    # Transform the input data
    input_data_scaled = scaler.transform(input_data)

    # Prediction
    prediction = classifier.predict(input_data_scaled)
    result = 'Above Threshold' if prediction[0] == 1 else 'Below Threshold'
    print(f'Prediction: {result}')

# Step 7: Main Script
def main():
    # Fetch data
    df = fetch_data_from_database()
    print("Initial Data:\n", df.head())

    # Preprocess data
    df = preprocess_data(df)
    print("Processed Data:\n", df.head())

    # Visualize data
    visualize_data(df)

    # Train and evaluate model
    classifier, scaler = train_decision_tree_model(df)

    # Predict for a new customer
    # Example: Predict for a customer with Age=45, Gender=1, Member_Card=2, Salary=97000
    predict_new_customer(classifier, scaler, 45, 1, 2, 97000)

if __name__ == "__main__":
    main()
