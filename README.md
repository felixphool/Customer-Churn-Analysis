# Customer Churn Prediction

## Project Overview  
This project aims to predict customer churn using machine learning techniques. By leveraging data analytics, we derive actionable insights to enhance customer retention strategies and improve business decision-making.

## Course  
**CSC522 Automated Learning and Data Analysis**  

## Dataset  
We used the **Telco Customer Churn dataset** from Kaggle:  
[Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  

- **7043** customer records  
- **21** features  
- **70-30%** train-test split  

### Key Features  
- **Personal Information**: Gender, age, dependents  
- **Service Details**: Phone, internet, security features  
- **Billing Information**: Monthly charges, payment methods, tenure  
- **Target Variable**: `Churn` (Binary - customer left or not)  

## Methodology  

### 1. Data Preprocessing  
- **Handling Missing Values**: Mean/mode imputation  
- **Feature Selection**: Dropped non-informative columns (e.g., `customerID`)  
- **Encoding & Scaling**:  
  - Label encoding for categorical variables  
  - Standard scaling for numeric values  
- **Class Imbalance Handling**: Used **SMOTE** to balance churn vs. non-churn classes  

### 2. Feature Engineering  
- Identified key features using heatmaps and domain knowledge  
- Created tenure-based customer groups  
- Engineered features to improve model interpretability  

### 3. Model Training & Evaluation  
Trained multiple machine learning models:  
- **Logistic Regression**  
- **Random Forest**  
- **Gradient Boosting**  
- **XGBoost**  
- **LightGBM**  
- **Decision Tree**  
- **KNN**  
- **Naive Bayes**  
- **SVM**  

#### Evaluation Metrics:  
- Accuracy  
- Precision, Recall, F1-score  
- AUC-ROC  

#### Best Performing Models  
- **Random Forest, LightGBM, Gradient Boosting** achieved around **85% accuracy** after hyperparameter tuning.

## Key Insights  
- Contract type, tenure, and payment methods are the most significant factors affecting churn.  
- Short-tenured customers are at the highest risk of churning.  
- Customers using automated payment methods are less likely to churn.  

## Future Scope  
- **Real-Time Prediction**: Deploy a model for real-time customer churn analysis.  
- **Automated Machine Learning**: Optimize models using AutoML techniques.  
- **Integration with Behavioral Data**: Enhance predictions by including customer interaction data.  

## Conclusion  
Machine learning enables businesses to predict customer churn with high accuracy. By implementing targeted retention strategies, companies can reduce attrition and improve customer loyalty.  
