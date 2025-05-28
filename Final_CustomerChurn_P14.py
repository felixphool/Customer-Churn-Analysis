#!/usr/bin/env python
# coding: utf-8

# <b>CSC 522 Automated Learning and Data Analysis</b><br>
# Project Group: 14 : {agmalpur}, {rkulkar5}, {radange}<br>
# Title: Customer Churn Prediction using Telco Customer Churn Dataset

# In[1]:


#importing all required libraries
import pyodbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import random
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import pickle

import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(2024)
random.seed(2024)


# <h2> Understanding the Data

# In[2]:


#Reading the Dataset from local source
df=pd.read_csv(r'C:\Users\Dell\Documents\GitHub\engr-ALDA-Fall2024-P14\Telco_Customer_Churn_with_NaNs.csv')
#Splitting the data into train and test sets (70% train data and 30% test data)
train, test = train_test_split(df, test_size=0.3, random_state=2024)
test=test.drop('Churn', axis=1) #dropping the 'Churn' column in test data to predict the results


# In[3]:


train.shape #checking the shape of the training data


# In[4]:


train.head() #glancing at the training data


# In[5]:


train.info() #getting insights upon the training data


# In[6]:


#getting the overall statistics for the training data 
train.describe(include='all').transpose()
#describe by default will only give values of numerical attributes, 'inlcude all' will also include categorical features.


# In[7]:


test.shape #getting the shape of the test dataset 


# In[8]:


test.head() #glancing over the test data


# In[9]:


test.info() #getting to learn about the test data by getting its information


# In[10]:


test.describe(include='all').transpose() #looking at the statistics of the test data


# In[11]:


#Checking for duplicates in the train dataset
train.duplicated().sum()


# In[12]:


#Checking for duplicates in the test dataset
test.duplicated().sum()


# In[13]:


#Finding the missing values in the train dataset
train.isna().sum()


# In[14]:


#Finding the missing values in the test dataset
test.isna().sum()


# In[15]:


#Splitting the features upon the target variable output and getting the count (training data)
print('OnlineSecurity: ')
print(train['OnlineSecurity'].value_counts())
print('\n Gender: ')
print(train['gender'].value_counts())
print('\n Senior Citizen: ')
print(train['SeniorCitizen'].value_counts())
print('\n Partner: ')
print(train['Partner'].value_counts())
print('\n Payment Method: ')
print(train['PaymentMethod'].value_counts())
print('\n Contract: ')
print(train['Contract'].value_counts())
print('\n Streaming Movies: ')
print(train['StreamingMovies'].value_counts())
print('\n Tech Support:')
print(train['TechSupport'].value_counts())
print('\n Paperless Billing: ')
print(train['PaperlessBilling'].value_counts())
print('\n Streamming TV: ')
print(train['StreamingTV'].value_counts())
print('\n Device Protection: ')
print(train['DeviceProtection'].value_counts())
print('\n Churn: ')
print(train['Churn'].value_counts())


# In[16]:


#Splitting the features upon the target variable output and getting the count (test data)
print('OnlineSecurity: ')
print(test['OnlineSecurity'].value_counts())
print('\n Gender: ')
print(test['gender'].value_counts())
print('\n Senior Citizen: ')
print(test['SeniorCitizen'].value_counts())
print('\n Partner: ')
print(test['Partner'].value_counts())
print('\n Payment Method: ')
print(test['PaymentMethod'].value_counts())
print('\n Contract: ')
print(test['Contract'].value_counts())
print('\n Streaming Movies: ')
print(test['StreamingMovies'].value_counts())
print('\n Tech Support:')
print(test['TechSupport'].value_counts())
print('\n Paperless Billing: ')
print(test['PaperlessBilling'].value_counts())
print('\n Streamming TV: ')
print(test['StreamingTV'].value_counts())
print('\n Device Protection: ')
print(test['DeviceProtection'].value_counts())


# In[17]:


#The number of categories in the target column (unique entries in the target column)
train['Churn'].unique()


# <h2> Data Visualization

# In[18]:


#The overall distribution of the categories in the Churn column in the training data visualized as a bar plot
sns.countplot(x='Churn', data=train)
plt.show()


# In[19]:


#dropping the customerID feature since it is a nominal attribute and does not contribute to the understanding of whether a customer may churn or not.
train.drop(['customerID'], axis=1, inplace=True)
test.drop(['customerID'], axis=1, inplace=True)
#dropping the customerID feature in both train and test datasets
train['SeniorCitizen'] = train['SeniorCitizen'].replace({0: 'No', 1: 'Yes'})
test['SeniorCitizen'] = test['SeniorCitizen'].replace({0: 'No', 1: 'Yes'})


# In[20]:


#plotting the histogram for numerical attributes to visualize the spread of values in it
train.hist(bins=20, figsize = (10,5))
plt.show
#TotalCharges is an object(ie. a categorical attribute) and therefore does not show up on one of these plots


# In[21]:


#plotting bar plots for all the categorical features in the dataset to visalize the overall spread of values in it

#categorical_columns stores all the categorical attributes in it, excluding attributes that have unique values in them (also the target variable)
categorical_columns = train.select_dtypes(include=['object', 'category']).columns
categorical_columns = categorical_columns.drop(['TotalCharges','Churn']) 
num_cols = 4  #We want 4 plots per row
num_rows = (len(categorical_columns) + num_cols - 1) // num_cols  #calculating the total rows required
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4)) #creating subplots
axes = axes.flatten() #flattening axes

# Plotting bar graphs for each categorical column in the dataset
for i, col in enumerate(categorical_columns):
    sns.countplot(data=train, x=col, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].tick_params(axis='x', rotation=45)

# Removing empty subplots
for i in range(len(categorical_columns), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# In[22]:


#Plotting bar graphs for the training dataset for attributes (to see the spread) according to the target variable Churn
num_cols = 4  #we require 4 plots per row
num_rows = (len(categorical_columns) + num_cols - 1) // num_cols  # Calculating the number of rows needed
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5)) # making subplots
axes = axes.flatten() #flattening axes

# Plot bar graphs for each categorical column with 'Churn' (depicted by the different hues)
for i, col in enumerate(categorical_columns):
    sns.countplot(data=train, x=col, hue='Churn', ax=axes[i])
    axes[i].set_title(f'{col} vs Churn')
    axes[i].tick_params(axis='x', rotation=45)
for i in range(len(categorical_columns), len(axes)): #removing the subplots
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# In[23]:


#Identifying the categorical columns (excluding 'Churn')
categorical_columns = train.select_dtypes(include=['object', 'category']).columns
#here we drop TotalCharges from the categorical ones since it is a numerical attribute and is currently treated as an object type. 
#We change it to float in the later part of the code
categorical_columns = categorical_columns.drop(['TotalCharges','Churn'])  # Exclude 'Churn'

#setting grid for visualizations
num_cols = 4
num_rows = (len(categorical_columns) + num_cols - 1) // num_cols  


fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 4)) 
axes = axes.flatten()  

# creating a stacked bar chart for all the categorical columns
for i, col in enumerate(categorical_columns):
    cross_tab = pd.crosstab(train['Churn'], train[col])
    cross_tab.plot(kind='bar', stacked=True, ax=axes[i], legend=False)  
    axes[i].set_title(f'Churn vs {col}', fontsize=10)
    axes[i].set_xlabel(col, fontsize=8)
    axes[i].set_ylabel('Count', fontsize=8)
    axes[i].tick_params(axis='x', labelrotation=45, labelsize=8) 

for j in range(len(categorical_columns), len(axes)):
    fig.delaxes(axes[j])

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Churn', loc='upper center', ncol=2, fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.suptitle('\n Churn vs Categorical Variables', fontsize=14)
plt.show()


# In[24]:


train['TotalCharges'] = pd.to_numeric(train['TotalCharges'], errors='coerce')

# Create the violin plot to better visualize the density trend of churn varying with total charges 
plt.figure(figsize=(10, 6))
sns.violinplot(x='Churn', y='TotalCharges', data=train, inner='quartile')
plt.title('Violin Plot of TotalCharges vs Churn')
plt.xlabel('Churn')
plt.ylabel('Total Charges')
plt.show()


# In[25]:


from statsmodels.graphics.mosaicplot import mosaic

# Creating a cross-tabulation of ISP and Churn
cross_tab = pd.crosstab(train['InternetService'], train['Churn'])

# Creating a mosaic plot for ISP vs Churn
plt.figure(figsize=(10, 6))
mosaic(cross_tab.stack(), title='Mosaic Plot of ISP vs Churn', labelizer=lambda k: f'{k[0]}: {k[1]}')
plt.xlabel('Internet Service')
plt.ylabel('Churn')
plt.show()


# In[26]:


import plotly.graph_objects as go

# Create a DataFrame to hold the counts for the Sankey diagram
sankey_data = train.groupby(['InternetService', 'Contract', 'PaymentMethod', 'Churn']).size().reset_index(name='Count')

# Create labels for the Sankey diagram
labels = (list(sankey_data['InternetService'].unique()) + 
          list(sankey_data['Contract'].unique()) + 
          list(sankey_data['PaymentMethod'].unique()) + 
          list(sankey_data['Churn'].unique()))

# Create a mapping for the labels to indices
label_index = {label: idx for idx, label in enumerate(labels)}

# Create source, target, and value lists for the Sankey diagram
sources = []
targets = []
values = []

# Populate the sources, targets, and values
for _, row in sankey_data.iterrows():
    internet_service_index = label_index[row['InternetService']]
    contract_index = label_index[row['Contract']] + len(sankey_data['InternetService'].unique())
    payment_method_index = (label_index[row['PaymentMethod']] + 
                            len(sankey_data['InternetService'].unique()) + 
                            len(sankey_data['Contract'].unique()))
    churn_index = (label_index[row['Churn']] + 
                   len(sankey_data['InternetService'].unique()) + 
                   len(sankey_data['Contract'].unique()) + 
                   len(sankey_data['PaymentMethod'].unique()))
    
    # First layer: InternetService to Contract
    sources.append(internet_service_index)
    targets.append(contract_index)
    values.append(row['Count'])
    
    # Second layer: Contract to PaymentMethod
    sources.append(contract_index)
    targets.append(payment_method_index)
    values.append(row['Count'])
    
    # Third layer: PaymentMethod to Churn
    sources.append(payment_method_index)
    targets.append(churn_index)
    values.append(row['Count'])

# Define colors for different categories
node_colors = []
for label in labels:
    if label in sankey_data['InternetService'].unique():
        node_colors.append('lightblue')  # InternetService color
    elif label in sankey_data['Contract'].unique():
        node_colors.append('lightgreen')  # Contract color
    elif label in sankey_data['PaymentMethod'].unique():
        node_colors.append('lightcoral')  # PaymentMethod color
    else:
        node_colors.append('green')  # Churn color

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        color=node_colors  # Apply colors here
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values
    ))])

fig.update_layout(title_text="Sankey Diagram: Flow of Internet Service, Contract Type, Payment Method, and Customer Churn", font_size=10)
fig.show()


# In[27]:


train['SeniorCitizen'] = train['SeniorCitizen'].replace({0: 'No', 1: 'Yes'})
test['SeniorCitizen'] = test['SeniorCitizen'].replace({0: 'No', 1: 'Yes'})
train['DeviceProtection'] = train['DeviceProtection'].replace({'No': 'No internet service'})
test['DeviceProtection'] = test['DeviceProtection'].replace({'No': 'No internet service'})
train['MultipleLines'] = train['MultipleLines'].replace({'No': 'No phone service'})
test['MultipleLines'] = test['MultipleLines'].replace({'No': 'No phone service'})
train['OnlineBackup'] = train['OnlineBackup'].replace({'No': 'No internet service'})
test['OnlineBackup'] = test['OnlineBackup'].replace({'No': 'No internet service'})

train['OnlineSecurity'] = train['OnlineSecurity'].replace({'No': 'No internet service'})
test['OnlineSecurity'] = test['OnlineSecurity'].replace({'No': 'No internet service'})
train['TechSupport'] = train['TechSupport'].replace({'No': 'No internet service'})
test['TechSupport'] = test['TechSupport'].replace({'No': 'No internet service'})
train['StreamingTV'] = train['StreamingTV'].replace({'No': 'No internet service'})
test['StreamingTV'] = test['StreamingTV'].replace({'No': 'No internet service'})
train['StreamingMovies'] = train['StreamingMovies'].replace({'No': 'No internet service'})
test['StreamingMovies'] = test['StreamingMovies'].replace({'No': 'No internet service'})


# In[28]:


#Change the data type of the Total Charges column
train['TotalCharges'] = train['TotalCharges'].replace(' ', np.nan)
#Converting The total charges column into float
train["TotalCharges"]=train.TotalCharges.apply(lambda x:float(x))


# In[29]:


#Change the data type of the Total charges column in the test data set
test['TotalCharges'] = test['TotalCharges'].replace(' ', np.nan)
#COnvert The total charges column into float
test["TotalCharges"]=test.TotalCharges.apply(lambda x:float(x))


# In[30]:


#The percentage of the missing values
(train.isna().sum()*100)/ len(train)


# ## Imputing the Missing Values

# In[31]:


#Imputing the numerical columns using mean
numerical_imputer = SimpleImputer(strategy='mean')


# In[32]:


#Numerical Columns
numerical_columns = ['tenure', 'TotalCharges','MonthlyCharges']


# In[33]:


#Imputing the numerical columns in the train dataset using fit transform
for column in numerical_columns:
    train[column] = numerical_imputer.fit_transform(train[column].values.reshape(-1, 1))
    test[column] = numerical_imputer.fit_transform(test[column].values.reshape(-1, 1))


# In[34]:


#To Impute the Categorical columns using mode as the replacement strategy
categorical_imputer = SimpleImputer(strategy='most_frequent')
categorical_columns = ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService' ,'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','Contract','PaperlessBilling','PaymentMethod' ,'Churn']
for column in categorical_columns:
    train[column] = categorical_imputer.fit_transform(train[column].values.reshape(-1, 1))
categorical_columns_test = ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService' ,'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','Contract','PaperlessBilling','PaymentMethod']
for column in categorical_columns_test:  
    test[column] = categorical_imputer.fit_transform(test[column].values.reshape(-1, 1))


# In[35]:


train.isna().sum()


# In[36]:


test.isna().sum()


# In[37]:


#Generating the heatmap to identify the correlated variables

label_encoded = train.copy()
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                       'PaperlessBilling', 'PaymentMethod', 'Churn']

# Use label encoding
for column in categorical_columns:
    label_encoded[column] = label_encoded[column].astype('category').cat.codes

# Compute the correlation matrix
correlation_matrix = label_encoded.corr()

plt.figure(figsize=(16, 14))

sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, 
            cbar_kws={"shrink": .8})

plt.title('Heatmap of Attribute Correlations Without Full Encoding')
plt.show()


# ## Experimental Design

# ### Research Questions

# In[38]:


# Count the number of churned customers
churned_customers = train[train['Churn'] == 'Yes'].shape[0]

# Count the total number of customers
total_customers = train.shape[0]

# Calculate the churn rate
churn_rate = (churned_customers / total_customers) * 100

print("Overall churn rate: {:.2f}%".format(churn_rate))


# In[39]:


#finding the relationship between the tenure of a customer and their likelihood of churn
#Grouping the data by tenure
tenure_groups = pd.cut(train['tenure'], bins=[0, 12, 24, 36, 48, 60, 72])
churn_by_tenure = train.groupby(tenure_groups)['Churn'].value_counts(normalize=True).unstack()
#Calculating the churn rate for each tenure group
churn_rate_by_tenure = churn_by_tenure['Yes']


# In[40]:


#Plotting the relationship
x = churn_rate_by_tenure.index.astype(str)
y = churn_rate_by_tenure.values

plt.bar(x, y)
plt.xlabel('Tenure Group')
plt.ylabel('Churn Rate')
plt.title('Churn Rate by Tenure Group')
plt.show()


# In[41]:


# Group the data by InternetService and calculate churn rate
churn_by_internet_service = train.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack()
churn_by_internet_service = churn_by_internet_service.loc[:, 'Yes']

# Plotting the churn rates for each internet service category
plt.figure(figsize=(8, 6))
sns.barplot(x=churn_by_internet_service.index, y=churn_by_internet_service.values)
plt.xlabel('Internet Service')
plt.ylabel('Churn Rate')
plt.title('Churn Rate by Internet Service')
plt.show()


# In[42]:


plt.figure(figsize=(10, 6))
sns.countplot(x='PaymentMethod', hue='Churn', data=train)
plt.xlabel('Payment Method')
plt.ylabel('Count')
plt.title('Churn Rates by Payment Method')
plt.show()


# In[43]:


# Create the contingency table
contingency_table = [[0.831924, 0.666886], [0.168076, 0.333114]]

# Perform the chi-square test
chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)

# Compare the p-value with the significance level
if p_value < 0.05:
    print("There is a significant relationship between the customer's internet service provider and their likelihood of churn.")
else:
    print("There is no significant relationship between the customer's internet service provider and their likelihood of churn.")


# ## Feature Engineering

# In[44]:


sns.heatmap(train.corr(),annot=True)


# ## Feature Encoding

# In[45]:


#Encoding the target column
# Perform label encoding on 'Churn' column
label_encoder = LabelEncoder()
train['Churn'] = label_encoder.fit_transform(train['Churn'])


# In[46]:


# Dividing numerical and categorical columns
numerical_columns = train.select_dtypes(include=['int64', 'float64','int32']).columns
categorical_columns = train.select_dtypes(include=['object']).columns

# Subset of the numerical and categorical columns
train_num = train[numerical_columns]
train_cat = train[categorical_columns]
# Convert boolean columns to strings
train_cat = train_cat.astype(str)
train_num.shape,train_cat.shape


# In[47]:


#Implementing OneHot Encoding
encoder = OneHotEncoder(sparse=False,drop='first',handle_unknown='error').set_output(transform='pandas')
encoder.fit(train_cat)
encoded_data = encoder.transform(train_cat)

encoded_data


# In[48]:


encoded_data.shape


# In[49]:


encoded_data.columns


# In[50]:


encoded_train = pd.concat([train_num, encoded_data], axis=1)
encoded_train


# ## Handling Data Imbalance
# 

# In[51]:


# Splitting the dataset into features (X) and target variable (y)
X = encoded_train.drop('Churn', axis=1)
y = encoded_train['Churn']


# In[52]:


encoded_data.describe().transpose()


# In[53]:


#Implementing SMOTE to help balance our data evenly in accordance to our class variable i.e Churn
smote = SMOTE(sampling_strategy='minority')
X_resampled, y_resampled= smote.fit_resample(X, y)
#Showcasing a equal count for each target value
y_resampled.value_counts()


# In[54]:


print(X_resampled.shape)
print(y_resampled.shape)


# ## Data Splitting

# In[55]:


# Split the data into training and evaluation sets (80-20 split ratio)
X_train, X_eval, y_train, y_eval = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42,stratify=y_resampled)


# In[56]:


X_train.shape,X_eval.shape,y_train.shape,y_eval.shape


# ### Data Scaling

# In[57]:


#Cloning the standardscaler class
scaler =StandardScaler().set_output(transform="pandas")
# Columns to be scaled
columns_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
#Fitting and transforming the selected columns in the training data set
X_train_scaled = X_train.copy()
X_train_scaled[columns_to_scale]=scaler.fit_transform(X_train[columns_to_scale])

#Transforming the selected columns  in the evaluation dataset
X_eval_scaled=X_eval.copy() 
X_eval_scaled[columns_to_scale] = scaler.transform(X_eval[columns_to_scale])


# In[58]:


#Checking on the scaled training set
X_train_scaled.head()


# In[59]:


#To check on the scaled evaluation on the predictor set
X_eval_scaled.head()


# ### Data Modelling

# In[60]:


# Defining a list of models that will be evaluated
models = [
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('Support Vector Machine', SVC(random_state=42)),
    ('Gaussian Naive Bayes', GaussianNB()),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('XGBoost', XGBClassifier(random_state=42)),
    ('LightGBM', LGBMClassifier(random_state=42)),
]

results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC'])

for model_name, model in models:
    # Fitting the models to the training dataset
    model.fit(X_train_scaled, y_train)

    # Make predictions using the evaluation dataset
    y_pred = model.predict(X_eval_scaled)

    accuracy = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred)
    recall = recall_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred)
    auc_roc = roc_auc_score(y_eval, y_pred)

    results_df = results_df.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC-ROC': auc_roc
    }, ignore_index=True)



# In[61]:


#Evaluation scores dataframe
results_df


# ## Hyperparameter Tuning
# 

# #### Random Forest Model

# In[62]:


RandomForestClassifier().get_params()


# In[63]:


# Defining the parameter grid for Random Forest m
param_grid_rf = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 50],
    'min_samples_split': [2, 5,10, 12],
    'min_samples_leaf': [1, 2,4, 8]
}


# In[64]:


# Defining a random forest model
rf = RandomForestClassifier(random_state=42)
#Perfoming a grid search using cross validation
grid_search_rf = GridSearchCV(rf, param_grid_rf, scoring='f1', cv=5)
grid_search_rf.fit(X_train_scaled, y_train)


# In[65]:


# Getting the best parameters
best_params_rf = grid_search_rf.best_params_
best_score_rf = grid_search_rf.best_score_


# In[66]:


#best parameters for the random forest model
print(f"The best hyperparameters for the Random Forest model are:\n{best_params_rf}")


# #### LightGBM model

# In[67]:


param_grid_lgbm = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.1, 0.01, 0.001]
}


# In[68]:


lgbm = LGBMClassifier(random_state=42)

# Performing Grid Search Cross Validation
grid_search_lgbm = GridSearchCV(lgbm, param_grid_lgbm, scoring='f1', cv=5)
grid_search_lgbm.fit(X_train_scaled, y_train)


# In[69]:


best_params_lgbm = grid_search_lgbm.best_params_
best_score_lgbm = grid_search_lgbm.best_score_
#Printing the best hyperparameters
print(f"The best hyperparameters for the light gbm model are:\n{best_params_lgbm}")


# #### Hyperparameter Tuning on the Catboost Model

# In[70]:


# Define the parameter grid for Grid Search
param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7]
}


# In[71]:


# Define the Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)


# In[72]:


# Performing Grid Search with cross-validation
grid_search = GridSearchCV(gb_model, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_scaled, y_train)


# In[73]:


# Get the best hyperparameters and the best model
best_params = grid_search.best_params_
best_gb = grid_search.best_estimator_ 


# In[74]:


print(f"The best hyperparameters for the Gradient boosting classifier are: \n{best_params}")


#   ### Perfomance Comparison after Hyperparameter Tuning

# In[75]:


results_tuned_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC'])

rf_model_tuned = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=1, random_state=42)
lgbm_model_tuned = LGBMClassifier(learning_rate=0.1, n_estimators=100, max_depth=5, random_state=42)
gb_model_tuned = GradientBoostingClassifier(random_state=42,learning_rate=0.1,max_depth=3,n_estimators=100)
# Fitting the tuned models to the training data
rf_model_tuned.fit(X_train_scaled, y_train)
lgbm_model_tuned.fit(X_train_scaled, y_train)
gb_model_tuned.fit(X_train_scaled,y_train)


# predictions on the evaluation data
y_pred_rf = rf_model_tuned.predict(X_eval_scaled)
y_pred_lgbm = lgbm_model_tuned.predict(X_eval_scaled)
y_pred_gb =gb_model_tuned.predict(X_eval_scaled)

# Calculating the evaluation scores of different models
accuracy_rf = accuracy_score(y_eval, y_pred_rf)
precision_rf = precision_score(y_eval, y_pred_rf)
recall_rf = recall_score(y_eval, y_pred_rf)
f1_rf = f1_score(y_eval, y_pred_rf)
auc_roc_rf = roc_auc_score(y_eval, y_pred_rf)

accuracy_lgbm = accuracy_score(y_eval, y_pred_lgbm)
precision_lgbm = precision_score(y_eval, y_pred_lgbm)
recall_lgbm = recall_score(y_eval, y_pred_lgbm)
f1_lgbm = f1_score(y_eval, y_pred_lgbm)
auc_roc_lgbm = roc_auc_score(y_eval, y_pred_lgbm)

accuracy_gb = accuracy_score(y_eval, y_pred_gb)
precision_gb = precision_score(y_eval, y_pred_gb)
recall_gb = recall_score(y_eval, y_pred_gb)
f1_gb = f1_score(y_eval, y_pred_gb)
auc_roc_gb = roc_auc_score(y_eval, y_pred_gb)

# Compiling said results into a dataframe
results_tuned_df = results_tuned_df.append({'Model': 'Random Forest (Tuned)',
                                'Accuracy': accuracy_rf,
                                'Precision': precision_rf,
                                'Recall': recall_rf,
                                'F1 Score': f1_rf,
                                'AUC-ROC': auc_roc_rf}, ignore_index=True)

results_tuned_df = results_tuned_df.append({'Model': 'LightGBM (Tuned)',
                                'Accuracy': accuracy_lgbm,
                                'Precision': precision_lgbm,
                                'Recall': recall_lgbm,
                                'F1 Score': f1_lgbm,
                                'AUC-ROC': auc_roc_lgbm}, ignore_index=True)

results_tuned_df = results_tuned_df.append({'Model': 'Gradient Boosting (Tuned)',
                                'Accuracy': accuracy_gb,
                                'Precision': precision_gb,
                                'Recall': recall_gb,
                                'F1 Score': f1_gb,
                                'AUC-ROC': auc_roc_gb}, ignore_index=True)


# In[76]:


#Evaluation scores for the tuned models after hyper parameter tuning
results_tuned_df


# In[77]:


#Evaluation scores for the untuned models without hyper parameter tuning
results_df


# ### Feature Importance
# 

# In[78]:


# Get feature importances from the Random Forest model to indicate the most relevant datapoints
feature_importances = rf_model_tuned.feature_importances_

# Create a dataframe to store the feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
feature_importance_df


# In[79]:


# Plotting the feature importances with respect to their importance
plt.figure(figsize=(10, 6))
plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()


# In[ ]:




