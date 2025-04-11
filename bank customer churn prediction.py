import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import plotly.express as px

# Set page config
st.set_page_config(page_title="Bank Customer Churn Prediction", layout="wide")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("Churn_Modelling.csv")
    data_clean = data.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
    return data_clean

data = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:", 
                          ["Data Overview", "Exploratory Analysis", "Feature Engineering", "Model Training", "Predictions"])

# Main content
st.title("Bank Customer Churn Prediction Dashboard")

if options == "Data Overview":
    st.header("Data Overview")
    
    st.subheader("First 5 rows of the dataset")
    st.write(data.head())
    
    st.subheader("Dataset Information")
    st.write(f"Number of rows: {data.shape[0]}")
    st.write(f"Number of columns: {data.shape[1]}")
    
    st.subheader("Missing Values")
    st.write(data.isnull().sum())
    
    st.subheader("Duplicate Rows")
    st.write(f"Number of duplicates: {data.duplicated().sum()}")

elif options == "Exploratory Analysis":
    st.header("Exploratory Data Analysis")
    
    # Churn distribution
    st.subheader("Churn Distribution")
    fig = px.pie(data, names='Exited', title='Customer Churn Distribution')
    st.plotly_chart(fig, use_container_width=True)
    
    # Numerical features distribution
    st.subheader("Numerical Features Distribution")
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    selected_num = st.selectbox("Select numerical feature to visualize:", num_cols)
    
    fig = px.histogram(data, x=selected_num, color='Exited', 
                      marginal="box", nbins=50, 
                      title=f"Distribution of {selected_num} by Churn Status")
    st.plotly_chart(fig, use_container_width=True)
    
    # Categorical features
    st.subheader("Categorical Features Analysis")
    cat_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
    selected_cat = st.selectbox("Select categorical feature to visualize:", cat_cols)
    
    fig = px.histogram(data, x=selected_cat, color='Exited', barmode='group',
                      title=f"{selected_cat} vs Churn Status")
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

elif options == "Feature Engineering":
    st.header("Feature Engineering")
    
    st.subheader("Feature Importance")
    st.write("""
    Based on the EDA, we can see that:
    - Age, Balance, and IsActiveMember seem to have significant impact on churn
    - Geography and Gender also show differences in churn rates
    - We'll preprocess these features for modeling
    """)
    
    st.subheader("Preprocessing Steps")
    st.write("""
    1. One-Hot Encoding for categorical features (Geography, Gender)
    2. Scaling for numerical features
    3. Handling class imbalance (we have about 20% churn rate)
    """)
    
    # Show preprocessing code
    st.code("""
    # Define preprocessing steps
    numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    categorical_features = ['Geography', 'Gender']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    """, language='python')

elif options == "Model Training":
    st.header("Model Training")
    
    st.subheader("Train-Test Split")
    test_size = st.slider("Select test size ratio:", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random state:", 42)
    
    # Preprocessing
    numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    categorical_features = ['Geography', 'Gender']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    
    X = data.drop('Exited', axis=1)
    y = data['Exited']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    st.write(f"Training set size: {X_train.shape[0]}")
    st.write(f"Test set size: {X_test.shape[0]}")
    
    # Model training
    st.subheader("Random Forest Classifier")
    n_estimators = st.slider("Number of trees:", 50, 500, 100, 50)
    max_depth = st.slider("Max depth:", 2, 20, 5, 1)
    
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            # Preprocess data
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
            model.fit(X_train_processed, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_processed)
            y_prob = model.predict_proba(X_test_processed)[:, 1]
            
            st.success("Model trained successfully!")
            
            # Metrics
            st.subheader("Model Performance")
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            
            st.write(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
            # Feature importance
            st.subheader("Feature Importance")
            feature_names = numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
            importances = model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                         title='Feature Importance', orientation='h')
            st.plotly_chart(fig, use_container_width=True)

elif options == "Predictions":
    st.header("Make Predictions")
    
    st.subheader("Enter Customer Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.number_input("Credit Score", 300, 850, 650)
        geography = st.selectbox("Geography", data['Geography'].unique())
        gender = st.selectbox("Gender", data['Gender'].unique())
        age = st.number_input("Age", 18, 100, 35)
        tenure = st.number_input("Tenure (years with bank)", 0, 15, 5)
    
    with col2:
        balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
        num_products = st.number_input("Number of Products", 1, 4, 2)
        has_credit_card = st.selectbox("Has Credit Card", [1, 0])
        is_active = st.selectbox("Is Active Member", [1, 0])
        salary = st.number_input("Estimated Salary", 0.0, 300000.0, 100000.0)
    
    # Load preprocessor and model (in a real app, you'd save and load these)
    if st.button("Predict Churn Probability"):
        # Preprocess the input
        input_data = pd.DataFrame([[credit_score, geography, gender, age, tenure, 
                                   balance, num_products, has_credit_card, 
                                   is_active, salary]],
                                 columns=data.columns[:-1])
        
        # Preprocess
        numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        categorical_features = ['Geography', 'Gender']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])
        
        # Fit preprocessor (in a real app, you'd load a pre-fitted one)
        _ = preprocessor.fit(data.drop('Exited', axis=1))
        
        # Train a model (in a real app, you'd load a pre-trained one)
        model = RandomForestClassifier(random_state=42)
        model.fit(preprocessor.transform(data.drop('Exited', axis=1)), data['Exited'])
        
        # Make prediction
        processed_input = preprocessor.transform(input_data)
        proba = model.predict_proba(processed_input)[0][1]
        
        st.subheader("Prediction Result")
        st.write(f"The probability that this customer will churn is: {proba:.2%}")
        
        if proba > 0.5:
            st.error("High churn risk - consider retention strategies!")
        else:
            st.success("Low churn risk")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("Bank Customer Churn Prediction Dashboard")
st.sidebar.markdown("Built with Streamlit")