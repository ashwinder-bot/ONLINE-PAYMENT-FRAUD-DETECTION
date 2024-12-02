import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def load_and_preprocess_data():
    try:
        # Load the dataset
        data = pd.read_csv("dataset.csv")
        
        # Data preprocessing
        print("Missing values:\n", data.isnull().sum())
        print("\nTransaction types distribution:\n", data["type"].value_counts())
        
        return data
    except FileNotFoundError:
        print("Error: dataset.csv not found in the current directory")
        return None

def create_transaction_visualization(data):
    type_counts = data["type"].value_counts()
    fig = px.pie(
        values=type_counts.values,
        names=type_counts.index,
        hole=0.5,
        title="Distribution of Transaction Types"
    )
    fig.show()

def prepare_model_data(data):
    # Encoding categorical variables
    type_mapping = {
        "CASH_OUT": 1,
        "PAYMENT": 2,
        "CASH_IN": 3,
        "TRANSFER": 4,
        "DEBIT": 5
    }
    
    data["type"] = data["type"].map(type_mapping)
    data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
    
    # Feature selection
    features = ["type", "amount", "oldbalanceOrg", "newbalanceOrig"]
    X = np.array(data[features])
    y = np.array(data[["isFraud"]])
    
    return X, y

def train_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42
    )
    
    # Train the model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    # Model evaluation
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    
    return model

def predict_fraud(model, transaction_data):
    prediction = model.predict(transaction_data)
    return "Fraud" if prediction[0] == 1 else "No Fraud"

def main():
    # Load and process data
    data = load_and_preprocess_data()
    if data is None:
        return
    
    # Create visualization
    create_transaction_visualization(data)
    
    # Prepare data for modeling
    X, y = prepare_model_data(data)
    
    # Train model
    model = train_model(X, y)
    
    # Example prediction
    sample_transaction = np.array([[4, 9000.60, 9000.60, 0.0]])
    result = predict_fraud(model, sample_transaction)
    print(f"\nPrediction for sample transaction: {result}")

if __name__ == "__main__":
    main()



