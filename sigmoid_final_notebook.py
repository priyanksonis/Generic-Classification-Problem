#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE


# In[2]:


X = pd.read_csv('train.csv')
y = pd.read_csv('train_churn_labels.csv')


# In[3]:


# Check missing values
missing_percentage = X.isnull().mean()

# Drop columns with more than 95% missing values
threshold = 0.95
columns_to_drop = missing_percentage[missing_percentage > threshold].index
X = X.drop(columns=columns_to_drop)


# In[4]:


# Summary statistics for numerical features
print("Summary Statistics:\n", X.describe())

# Summary statistics for categorical features
print("Categorical Features Description:\n", X.describe(include='object'))


# In[5]:


X.head(20)


# In[6]:


y.head(10)


# In[7]:


# Check data types and missing values
print("Data Types:\n", X.dtypes)
print("\nMissing Values:\n", X.isnull().sum())


# In[8]:


# Identify Numerical and Categorical Columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns


# In[9]:


numerical_features


# In[10]:


categorical_features


# In[ ]:





# # Univariate Analysis
# 

# In[13]:


# Histogram for numerical features
for column in X.select_dtypes(include='number').columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(X[column], kde=True, bins=30)
    plt.title(f'Distribution of {column}')
    plt.show()


# In[13]:





# In[14]:


# Plot histograms for numerical features
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
X[numerical_features].hist(figsize=(15, 10), bins=20, edgecolor='black')
plt.suptitle('Distribution of Numerical Features')
plt.show()


# In[14]:





# In[18]:


# Box plot for numerical features
for column in X.select_dtypes(include='number').columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=X[column])
    plt.title(f'Box Plot of {column}')
    plt.show()


# In[18]:





# In[19]:


# Count plot for categorical features
for column in X.select_dtypes(include='object').columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=X[column], order=X[column].value_counts().index)
    plt.title(f'Count Plot of {column}')
    plt.show()


# In[19]:





# # Bivariate Analysis
# 

# In[20]:


# # Scatter plot for numerical features
# sns.pairplot(X.select_dtypes(include='number'))
# plt.show()


# In[20]:





# In[21]:


# Correlation Matrix
plt.figure(figsize=(12, 8))
correlation_matrix = X[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[22]:


# Box plot for numerical vs. categorical features
target_variable = 'target_variable'  # Replace with your target column

for column in X.select_dtypes(include='number').columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=y.squeeze(), y=X[column])
    plt.title(f'{column} by {target_variable}')
    plt.show()


# In[22]:





# In[23]:


# sns.pairplot(X[['Var109', 'Var112']])
# plt.show()


# In[24]:


len(numerical_features), len(categorical_features)


# In[25]:


# # Grouped box plot
# sns.boxplot(x=categorical_features, y=numerical_features, hue=y, data=X)
# plt.title('Grouped Box Plot')
# plt.show()


# # Outlier Detection

# In[26]:


# Box plot for outlier detection
for column in X.select_dtypes(include='number').columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=X[column])
    plt.title(f'Box Plot for Outlier Detection in {column}')
    plt.show()


# In[27]:


# Z-score method for detecting outliers
from scipy.stats import zscore

# Calculate Z-scores for each feature
z_scores = X.select_dtypes(include='number').apply(zscore)

# Identify outliers based on Z-score threshold
outliers = (z_scores > 3) | (z_scores < -3)
print(outliers.sum())


# In[28]:


outliers


# In[28]:





# In[28]:





# # Target Variable Analysis

# In[29]:


# Distribution of the target variable
plt.figure(figsize=(8, 4))
sns.histplot(y, kde=True)
plt.title('Distribution of Target Variable')
plt.show()



# In[30]:


# Box plot of numerical features against the target variable
for column in X.select_dtypes(include='number').columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=y.squeeze(), y=X[column])
    plt.title(f'{column} by Target Variable')
    plt.show()


# In[30]:





# In[30]:





# In[13]:


from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from scipy.sparse import issparse

class CorrelationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        # Ensure X is a DataFrame
        if issparse(X):
            X = pd.DataFrame.sparse.from_spmatrix(X)
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        # Compute the correlation matrix
        corr_matrix = X.corr()
        # Identify features to drop
        self.features_to_drop_ = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > self.threshold:
                    colname = corr_matrix.columns[i]
                    self.features_to_drop_.add(colname)
        return self

    def transform(self, X):
        # Ensure X is a DataFrame
        if issparse(X):
            X = pd.DataFrame.sparse.from_spmatrix(X)
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        return X.drop(columns=self.features_to_drop_)

    def fit_transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns
        elif isinstance(X, np.ndarray):
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
        elif issparse(X):
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
        return self.fit(X, y).transform(X)




# Preprocessing for Numerical Data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for Categorical Data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# Combine Preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the Model
model = RandomForestClassifier(random_state=42)

# Create the Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('selector', CorrelationSelector(threshold=0.8)),  # Drop highly correlated features
    ('classifier', model)
])

# # Evaluate Pipeline with 10-Fold Cross-Validation
# scores = cross_val_score(pipeline, X, y, cv=10, scoring='accuracy')



# StratifiedKFold for Handling Class Imbalance
stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Evaluate Pipeline with Stratified Cross-Validation
scores = cross_val_score(pipeline, X, y, cv=stratified_kfold, scoring='accuracy', error_score='raise')


# Print Cross-Validation Scores
print("Cross-Validation Accuracy Scores:", scores)
print("Mean Accuracy:", scores.mean())

# Fit the Pipeline on the Entire Dataset
pipeline.fit(X, y)

# Example Predictions (optional)
# predictions = pipeline.predict(X)


# In[ ]:





# In[22]:


from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from scipy.sparse import issparse

class CorrelationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        # Ensure X is a DataFrame
        if issparse(X):
            X = pd.DataFrame.sparse.from_spmatrix(X)
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        # Compute the correlation matrix
        corr_matrix = X.corr()
        # Identify features to drop
        self.features_to_drop_ = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > self.threshold:
                    colname = corr_matrix.columns[i]
                    self.features_to_drop_.add(colname)
        return self

    def transform(self, X):
        # Ensure X is a DataFrame
        if issparse(X):
            X = pd.DataFrame.sparse.from_spmatrix(X)
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        return X.drop(columns=self.features_to_drop_)

    def fit_transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns
        elif isinstance(X, np.ndarray):
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
        elif issparse(X):
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
        return self.fit(X, y).transform(X)




# Preprocessing for Numerical Data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for Categorical Data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# Combine Preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the Model
model = RandomForestClassifier(random_state=42)

# Create the Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('selector', CorrelationSelector(threshold=0.8)),  # Drop highly correlated features
    ('classifier', model)
])

# # Evaluate Pipeline with 10-Fold Cross-Validation
# scores = cross_val_score(pipeline, X, y, cv=10, scoring='accuracy')

# StratifiedKFold for Handling Class Imbalance
stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Evaluate Pipeline with Stratified Cross-Validation for Multiple Metrics
scoring = {
    'accuracy': 'accuracy',
    'f1': make_scorer(f1_score),
    'roc_auc': 'roc_auc',
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score)
}

scores = cross_validate(pipeline, X, y, cv=stratified_kfold, scoring=scoring, error_score='raise')

# Print Cross-Validation Scores
print("Cross-Validation Scores:")
for metric in scoring.keys():
    print(f"{metric.capitalize()} Scores: {scores['test_' + metric]}")
    print(f"Mean {metric.capitalize()}: {scores['test_' + metric].mean()}")
    print(f"Standard Deviation {metric.capitalize()}: {scores['test_' + metric].std()}\n")

# Fit the Pipeline on the Entire Dataset
pipeline.fit(X, y)

# Example Predictions (optional)
# predictions = pipeline.predict(X)

