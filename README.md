# Bank-Customer-Churn-Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-ff69b4.svg)

##  Project Overview

This project develops a predictive model to identify bank customers at risk of churning (discontinuing services). The solution helps banks proactively implement retention strategies to:
- Reduce revenue loss
- Lower customer acquisition costs
- Maintain market share

##  Objectives

1. **Identify churn drivers**: Discover key factors influencing customer attrition
2. **Predict high-risk customers**: Generate probability scores for near-future churn
3. **Recommend interventions**: Suggest personalized retention strategies

##  Dataset

Contains 10,000 customer profiles with 11 features:

| Feature           | Description                                      | Type          |
|-------------------|--------------------------------------------------|---------------|
| CreditScore       | Numerical credit assessment (300-850)            | Integer       |
| Geography         | Customer location (France, Germany, Spain)       | Categorical   |
| Gender            | Male/Female                                      | Binary        |
| Age               | Customer age in years                            | Integer       |
| Tenure            | Years as bank customer                           | Integer       |
| Balance           | Account balance                                  | Float         |
| NumOfProducts     | Number of bank products used                     | Integer       |
| HasCrCard         | Credit card ownership (1=Yes, 0=No)              | Binary        |
| IsActiveMember    | Active usage status (1=Active, 0=Inactive)       | Binary        |
| EstimatedSalary   | Approximate annual salary                        | Integer       |
| Exited            | Churn status (1=Churned, 0=Retained)             | Binary        |

##  Technical Implementation

### Workflow

1. **Data Cleaning**
   - Removed non-predictive columns (RowNumber, CustomerId, Surname)
   - Handled outliers in Age, CreditScore, and NumOfProducts
   - Verified no missing values

2. **Exploratory Data Analysis (EDA)**
   - Analyzed feature distributions and correlations
   - Visualized relationships between features and churn

3. **Feature Engineering**
   - Encoded categorical variables
   - Scaled numerical features
   - Created interaction terms

4. **Model Training**
   - Evaluated multiple algorithms:
     - Logistic Regression
     - Random Forest
     - K-Nearest Neighbors
     - Gaussian Naive Bayes
   - Optimized hyperparameters using GridSearchCV

5. **Model Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
   - Confusion matrix analysis
   - Feature importance interpretation

##  Key Insights

- **Top churn drivers**: 
  - Low activity (IsActiveMember=0)
  - Higher age customers
  - Specific geographic regions
  - Lower product engagement

- **Retention recommendations**:
  - Targeted reactivation campaigns
  - Age-specific product bundles
  - Regional service improvements
  - Cross-selling strategies
    
- **Deployed Streamlit Dashboard**
   [Bank Cuatomer Churn Dashboard](https://leboswaratlhe-bank-customer--bankcustomerchurnprediction-sw8vcj.streamlit.app/)
  
##  How to Run

1. Clone repository:
   ```bash
   git clone [github](https://github.com/LeboSwaratlhe/Bank-Customer-Churn-Prediction.git)
   cd bank-churn-prediction

