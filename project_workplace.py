import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ------------------ Streamlit Config ------------------
st.set_page_config(page_title="Bank Churn Prediction", layout="centered")

st.title("🏦 Bank Customer Churn Prediction App")

# ------------------ Load and Clean Data ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Churn_Modelling.csv")
    
    # Feature engineering
    df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['TenureByAge'] = df['Tenure'] / (df['Age'] + 1)
    df['CreditScoreGivenAge'] = df['CreditScore'] / (df['Age'] + 1)
    
    # One-hot encoding
    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
    
    return df

data = load_data()
st.success("✅ Data loaded successfully!")

# ------------------ Feature & Target Split ------------------
target = 'Exited'
features = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
    'BalanceSalaryRatio', 'TenureByAge', 'CreditScoreGivenAge',
    'Geography_Germany', 'Geography_Spain', 'Gender_Male'
]

X = data[features]
y = data[target]

# ------------------ Train-Test Split & Scaling ------------------
@st.cache_resource
def prepare_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

if st.button("Train Model"):
    with st.spinner("Training model..."):
        model, scaler = prepare_model(X, y)
        joblib.dump(model, "rf_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
    st.success("🎉 Model trained and saved!")

# ------------------ Prediction Section ------------------
st.subheader("🔍 Predict Churn")

with st.form("prediction_form"):
    credit_score = st.slider("Credit Score", 300, 900, 650)
    age = st.slider("Age", 18, 100, 35)
    tenure = st.slider("Tenure", 0, 10, 3)
    balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
    num_products = st.slider("Number of Products", 1, 4, 1)
    estimated_salary = st.number_input("Estimated Salary", 1000.0, 200000.0, 50000.0)
    geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
    gender = st.selectbox("Gender", ['Female', 'Male'])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Load model and scaler
    try:
        model = joblib.load("rf_model.pkl")
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        st.error("❌ Please train the model first.")
    else:
        # Feature engineering for input
        BalanceSalaryRatio = balance / (estimated_salary + 1)
        TenureByAge = tenure / (age + 1)
        CreditScoreGivenAge = credit_score / (age + 1)
        
        input_data = {
            'CreditScore': credit_score,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'EstimatedSalary': estimated_salary,
            'BalanceSalaryRatio': BalanceSalaryRatio,
            'TenureByAge': TenureByAge,
            'CreditScoreGivenAge': CreditScoreGivenAge,
            'Geography_Germany': 1 if geography == 'Germany' else 0,
            'Geography_Spain': 1 if geography == 'Spain' else 0,
            'Gender_Male': 1 if gender == 'Male' else 0
        }

        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.write(f"🔮 Churn Prediction: {'Yes' if prediction else 'No'}")
        st.write(f"📊 Probability of Churn: {probability:.2%}")
