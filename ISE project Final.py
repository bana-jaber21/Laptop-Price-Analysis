#!/usr/bin/env python
# coding: utf-8

# ## Group 4
# 
# ### Problem Definition
# #### Our goal of this project is to clean and prepare a laptop dataset for further analysis or machine learning.
# #### The dataset includes information about laptop specifications such as Brand, Model, RAM, Screen size, Storage type, GPU, and Final Price. 
# 
# #### The main objectives are:
# 
# - Clean the data by fixing missing values and errors.
# - Understand how key features (like RAM, Screen size, and Price) are related to help choose important features.
# - Prepare the data for further tasks, such as predicting laptop prices or grouping similar laptops together.

# In[166]:


# Read the data and display them
import pandas as pd
df = pd.read_csv('laptops.csv', delimiter=',')
display(df)
# dispaly the data information
df.info()


# In[167]:


# To show any missing value
print("\nMissing Data:")
print(df.isnull().sum())


# ### Data Collection
# #### The dataset, laptops.csv, has 2160 rows and 12 columns. 
# #### It contains both numerical and categorical data:
# 
# - Categorical Features: Brand, Model, CPU, GPU, Storage type, Touch, Status.
# - Numerical Features: RAM, Screen size, Storage, Final Price.
# - Target Variable: The target variable for this analysis is Final Price, which shows the price of the laptop.
# 
# #### Some columns had missing values:
# - GPU had the most missing data (1371 missing entries).
# - Brand, Storage type, and RAM had fewer missing values.
# - The Final Price column contained negative values, which needed to be corrected.

# ### Feature Selection and Data Preprocessing
# 
# #### The following steps were used to clean and prepare the data:
# 

# In[168]:


# Fill missing values in 'RAM' with its median (due the outliers)
df['RAM'].fillna(df['RAM'].median(), inplace=True)

# Fill missing values in 'Screen' with its median (due the outliers)
df['Screen'].fillna(df['Screen'].median(), inplace=True)


# In[169]:


# Fill missing values in 'Brand' with its mode
df['Brand'].fillna(df['Brand'].mode()[0], inplace=True)

# Fill missing values in 'Storage type' with its mode
df['Storage type'].fillna(df['Storage type'].mode()[0], inplace=True)


# In[170]:


# Fill missing 'GPU' values with 'None' instead of dropping them, to avoid losing a huge number of data
df['GPU'] = df['GPU'].fillna('None')


# #### Handling Missing Values:
# - Numerical Features (RAM, Screen): Missing values were filled with the median to avoid the impact of outliers.
# - Categorical Features (Brand, Storage type): Missing values were filled with the mode, which is the most common category.
# - GPU: Missing GPU entries were replaced with "None" to prevent losing too much data.

# In[172]:


#Listing the negative values in price
print("\nNegative values in 'Price' column:")
negative_values = df['Final Price'][df['Final Price'] < 0]
print(negative_values)

#Applying absolte value on negative prices
df["Final Price"]=df["Final Price"].apply(lambda x: abs(x) if x<0 else x)

      
#Listing outliers in Screen 
Q1 = df['Screen'].quantile(0.25)
Q3 = df['Screen'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
#outliers = df[(df['Screen'] < lower_bound) | (df['Screen'] > upper_bound)]
outliers = df['Screen'][(df['Screen'] < lower_bound) | (df['Screen'] > upper_bound)]
print("\nOutliers in 'Screen' column:")
print(outliers)

      
#Dropping rows with outliers in Screen
df = df[(df['Screen'] >= lower_bound) & (df['Screen'] <= upper_bound)]


# #### Handling Negative Prices:
# - Final Price: Negative values were found and converted to positive values to fix any data errors.
# 

# #### Outlier Detection and Treatment:
# - Screen size: Outliers (values far from the norm) were identified using the Interquartile Range (IQR) method and removed to improve the data's reliability.

# In[173]:


display(df)
print(df.isnull().sum())


# ### Exploratory Data Analysis (EDA):

# In[174]:


# Statistical summary for numerical data
print('Statistical summary for numerical:')
numericdata = df.select_dtypes(include="number")
nsummary = numericdata.describe()  # summary for numeric data
nsummary.loc['median'] = numericdata.median()  # add median
nsummary.loc['mode'] = numericdata.mode().iloc[0]  # add mode
nsummary.loc['range'] = numericdata.max() - numericdata.min()  # add range
print(nsummary)

# Statistical summary for categorical data
print('\nStatistical summary for categorical:')
categoricalcolumns = df.select_dtypes(include=['object'])
#using loop to get each colum summry
for column in categoricalcolumns.columns:
    freq = categoricalcolumns[column].value_counts() #get frequncy
    percent = categoricalcolumns[column].value_counts(normalize=True) * 100 #the percentege
    mode = categoricalcolumns[column].mode()[0] if not categoricalcolumns[column].mode().empty else None #the mode
    #add them in summary:
    summary = pd.DataFrame({
        'Frequency': freq,
        'Percentage': percent
    })
    print(f"\nSummary for column: {column}")
    print(summary)
    #tell the mode of each colume
    print(f"Mode: {mode}")


# ##### Statistical Summary for Numerical Data:
# - A statistical summary was generated for the numerical columns, including measures of central tendency (mean, median, mode) and dispersion (standard deviation, range). 
# - This provided insights into the distribution and variability of the numerical features.
# 
# ##### Statistical Summary for Categorical Data:
# - The categorical columns were analyzed to identify the frequency and percentage of each category. 
# - The mode for each column was also determined to highlight the most common category.

# #### Distribution of Features:

# In[175]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 1. Histogram for counts of laptops by brand (Categorical)
plt.figure(figsize=(20, 10))
sns.countplot(x='Brand', data=df, order=df['Brand'].value_counts().index)
plt.title('Count of Laptops by Brand')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
########################### WE CAN IGNORE IT SINCE IT DOES NOT GIVE VALUABLE INFORMATION 
########################### ALSO, THE BRANDS ARE A LOT AND IT IS HARD TO CONVERT INTO NUMERICAL


# In[176]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Brand', y='Final Price', data=df)
plt.title('Final Price by Brand')
plt.xlabel('Brand')
plt.ylabel('Final Price ($)')
plt.xticks(rotation=45)
plt.grid()
plt.show()


# In[177]:


numerical_columns = ['RAM', 'Storage', 'Screen', 'Final Price']  # Add other numerical columns if necessary
categorical_columns = ['Brand', 'CPU', 'Storage type', 'Touch']  # Include relevant categorical columns
# exploring the correlation between numerical values
sns.pairplot(df[numerical_columns])
plt.title('Pair Plot of Numerical Features')
plt.show()


# In[179]:


# To differentiate by Storage type 
sns.pairplot(df[numerical_columns + ['Storage type']], hue='Storage type')
plt.title('Pair Plot of Numerical Features by Storage Type')
plt.show()


# In[180]:


# To differentiate by Touch
sns.pairplot(df[numerical_columns + ['Touch']], hue='Touch')
plt.title('Pair Plot of Numerical Features by Touch')
plt.show()


# In[181]:


# 2. Histograms for numerical features ( COUNT OF RAM AND FINAL PRICE )
num_columns = ['RAM', 'Final Price']
for col in num_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f'Histogram of {col}')
    plt.show()


# In[182]:


plt.figure(figsize=(8, 6))
sns.countplot(y='Screen', data=df)
plt.title('Count Plot of {Screen}')
plt.show()
########################################
plt.figure(figsize=(8, 6))
sns.countplot(y='Status', data=df)
plt.title('Count Plot of {Status}')
plt.show()


# In[183]:


########################################################################################
# Calculate the correlation matrix
correlation = df.corr(numeric_only=True)

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Create a heatmap for the correlation matrix
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()


# In[184]:


# 2. Scatter plot of Final Price vs RAM (numerical)
plt.figure(figsize=(8, 20))
sns.scatterplot(x='RAM', y='Final Price', data=df, hue='Status', alpha=0.7)
plt.title('Final Price vs RAM')
plt.show()


# In[185]:


sns.histplot(data=df, x='RAM', y='Final Price', bins=30, pmax=0.99, cmap='Blues')
plt.title('Bivariate Histogram of Final Price vs RAM')
plt.xlabel('RAM (GB)')
plt.ylabel('Final Price')
plt.show()


# ##### Histograms and count plots were used to visualize the distribution of numerical and categorical variables: 
# - RAM and Final Price had skewed distributions, meaning most laptops had lower prices and less RAM.
# - Examining Brands and Status helped identify trends in the most common laptop types.
# 
# #### Outliers:
# - Boxplots showed potential outliers, especially for Screen size and Final Price.
# 
# #### Correlation Heatmaps identified key relationships:
# - RAM and Final Price showed a positive correlation, meaning laptops with more RAM tend to have a higher price.
# - Storage and Screen size also had a relationship with Final Price, suggesting these features could help predict laptop prices.
# 
# #### Relationships Between Features:
# - Scatter Plots and Pair Plots showed how features like RAM and Screen size were related to Final Price.
# 
# #### Correlation Heatmaps identified key relationships:
# - RAM and Final Price showed a positive correlation, meaning laptops with more RAM tend to have a higher price.
# - Storage and Screen size also had a relationship with Final Price, suggesting these features could help predict laptop prices.

# In[186]:


# get the unique value in each column
unique_counts = df.nunique()
print(unique_counts)


# #### Encoding Categorical Variables:
# - One-Hot Encoding: Applied to features like Brand, Model, CPU, and GPU to turn them into numbers that can be used by machine learning algorithms.
# - Label Encoding: Applied to binary features (e.g., Touch, Storage type, Status) to convert them into 0 or 1.
# 

# In[188]:


from sklearn.preprocessing import LabelEncoder
# Apply one-hot encoding to categorical columns
df_encoded = pd.get_dummies(df, columns=['Brand', 'Model', 'CPU', 'GPU'], drop_first=True)

# Initialize LabelEncoder for 'Touch','Storage type', and 'Status'
label_encoder = LabelEncoder()

# Apply LabelEncoder to 'Touch'
df_encoded['Touch'] = label_encoder.fit_transform(df['Touch'])

# Apply LabelEncoder to 'Status'
df_encoded['Status'] = label_encoder.fit_transform(df['Status'])

# Apply LabelEncoder to 'Storage type'
df_encoded['Storage type'] = label_encoder.fit_transform(df['Storage type'])

# Display the transformed DataFrame
display(df_encoded.head())


# In[189]:


# Drop the 'Laptop' column as it is not useful for regression
df_encoded = df_encoded.drop('Laptop', axis=1)

# Display the updated DataFrame
display(df_encoded)


# #### Feature Removal:
# - The Laptop column was removed because it wasn’t useful for predicting laptop prices.

# ### Conclusion
# #### Up to this phase we successfully cleaned, processed, and analyzed the laptop dataset, preparing it for future machine learning tasks. 
# #### Our main steps included filling missing values, fixing negative prices, removing outliers, visualizing the data, and encoding categorical features into numerical ones.

# ### Data Preprocessing and Splitting:

# In[190]:


# Get a statistical summary of the dataframe (transpose it for easier reading)
df_encoded.describe().T


# In[217]:


# Generate Train - Test splits
from sklearn.model_selection import train_test_split

# Selecting all columns except the 6th column as features (X)
X = df_encoded.iloc[:, df_encoded.columns != df_encoded.columns[6]]  # Exclude the column at index 6 as target

# Set 'Final Price' as the target variable (y)
y = df_encoded['Final Price'].values  # Final price column as target variable

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# In[218]:


# Scaling the Train - Test splits to normalize the data
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler for data normalization
scaler = StandardScaler()

# Fit the scaler on the training data (X_train + y_train combined)
scaler.fit(np.c_[X_train, y_train])  # Scaling the combined data (features + target)

# Apply the scaling transformation to the training data
A_train = scaler.transform(np.c_[X_train, y_train])  # Apply scaling transformation
X_train = A_train[:, :-1]  # Select all columns except the last one as features (X_train)
y_train = A_train[:, -1]  # Select the last column as target (y_train)

# Apply the scaling transformation to the test data
A_test = scaler.transform(np.c_[X_test, y_test])  # Apply scaling transformation
X_test = A_test[:, :-1]  # Select all columns except the last one as features (X_test)
y_test = A_test[:, -1]  # Select the last column as target (y_test)

# Print the scaled training data for review
print(A_train)


# #### Dataset Splitting:
# - The dataset was divided into 80% training data and 20% testing data using train_test_split from scikit-learn.
# 
# #### Target Variable:
# - The target variable, "Final Price", was separated from the feature columns.
# 
# #### Data Scaling:
# - The feature data was standardized using StandardScaler to ensure all features have the same scale and contribute equally to the model.
# - Both the training and testing datasets were scaled before model training.
# 
# #### Model Readiness:
# - After scaling, the data was prepared for model training and evaluation, with the target variable separated out for future predictions.

# ### Regression Analysis

# In[219]:


# Regression Analysis: Mean Squared Error Metric
from sklearn.metrics import mean_squared_error

## OLS
from sklearn.linear_model import LinearRegression
reg1=LinearRegression(fit_intercept= False).fit(X_train, y_train)
Y_pred1 = reg1.predict(X_test)
print("The MSE using OLS is:", mean_squared_error(y_test, Y_pred1))


## Ridge
from sklearn.linear_model import RidgeCV
reg2=RidgeCV(alphas=[1e-3, 1e-2 , 1e-1 , 1e0 , 1e1 , 1e2 , 1e3], fit_intercept= False,cv=10).fit(X_train, y_train)
Y_pred2 = reg2.predict(X_test)
print("The MSE using RIDGE is:", mean_squared_error(y_test, Y_pred2))


## Lasso
from sklearn.linear_model import LassoCV
reg3 = LassoCV(alphas=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], 
               fit_intercept=False, cv=10, random_state=0).fit(X_train, y_train)
Y_pred3 = reg3.predict(X_test)
print("The MSE using Lasso is:", mean_squared_error(y_test, Y_pred3))


# #### based on the best values for the estimates are the values optained by Lasso since it performs better than Ridge and OLS.
# 

# In[220]:


print(" The best penalty coefficient:", reg3.alpha_)
reg3_coef = np.round(reg3.coef_,2)
print(" The best coefficient estimates are:", reg3_coef.tolist())


# ### Summary 
# Our team developed a regression model to predict laptop prices based on various features extracted from a dataset. We performed data preprocessing, including cleaning missing values, encoding categorical variables, removing outliers, and scaling features for consistency. Afterward, we explored the dataset using visual and statistical analysis to gain insights into feature distributions and relationships. This included generating count plots, histograms, pair plots, box plots, and a heatmap showing feature correlations.
# We then split the dataset into training and testing sets, applied multiple regression models (OLS, Ridge, and Lasso), and evaluated their performance using Mean Squared Error (MSE). Among the models, Lasso Regression provided the best results due to its feature selection capability and lower MSE, making it the most suitable model for our task.

# ### Communicate results
# - OLS Regression: The MSE was extremely high (1.85 × 10²⁵), indicating significant prediction errors. This result suggests that the OLS model was highly sensitive to multicollinearity and lacked regularization, making it unsuitable.
# 
# - Ridge Regression: The MSE was 0.1170, showing a notable improvement. Ridge reduced overfitting, controlling large feature coefficients without eliminating them entirely.
# 
# - Lasso Regression: The best-performing model was Lasso Regression, with an MSE of 0.1156. Lasso not only reduced overfitting but also performed automatic feature selection by shrinking less important coefficients to zero.
# - The results indicate that Lasso Regression is the most suitable model for predicting laptop prices. 

# ### Operationalize 
# To adopt our methodology, follow these steps:
#    - Set up the development environment
#    - Load and preprocess data
#    - Train the model
#    - Evaluate the model
#    - Make predictions for new laptops: Input New Laptop Features, preprocess the new input by encoding and scaling, then predict price
#    - Future improvements could involve adding more relevant features to improve the model's performance. 
