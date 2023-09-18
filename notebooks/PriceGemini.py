#!/usr/bin/env python
# coding: utf-8

# # PriceGemini

# ### Introduction About the Data :
# 
# **The dataset** The goal is to predict `price` of given diamond (Regression Analysis).
# 
# There are 10 independent variables (including `id`):
# 
# * `id` : unique identifier of each diamond
# * `carat` : Carat (ct.) refers to the unique unit of weight measurement used exclusively to weigh gemstones and diamonds.
# * `cut` : Quality of Diamond Cut
# * `color` : Color of Diamond
# * `clarity` : Diamond clarity is a measure of the purity and rarity of the stone, graded by the visibility of these characteristics under 10-power magnification.
# * `depth` : The depth of diamond is its height (in millimeters) measured from the culet (bottom tip) to the table (flat, top surface)
# * `table` : A diamond's table is the facet which can be seen when the stone is viewed face up.
# * `x` : Diamond X dimension
# * `y` : Diamond Y dimension
# * `x` : Diamond Z dimension
# 
# Target variable:
# * `price`: Price of the given Diamond.
# 
# Dataset Source Link :
# [https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv](https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv)

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import xgboost
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics


# ### Feature Description
# 
# * `price` price in US dollars ($326--$18,823)This is the target column containing tags for the features. 
# 
# **The 4 Cs of Diamonds:**
# 
# * `carat (0.2--5.01)` The carat is the diamond’s physical weight measured in metric carats.  One carat equals 1/5 gram and is subdivided into 100 points. Carat weight is the most objective grade of the 4Cs. 
# 
# * `cut (Fair, Good, Very Good, Premium, Ideal)` In determining the quality of the cut, the diamond grader evaluates the cutter’s skill in the fashioning of the diamond. The more precise the diamond is cut, the more captivating the diamond is to the eye.  
# 
# * `color, from J (worst) to D (best)` The colour of gem-quality diamonds occurs in many hues. In the range from colourless to light yellow or light brown. Colourless diamonds are the rarest. Other natural colours (blue, red, pink for example) are known as "fancy,” and their colour grading is different than from white colorless diamonds.  
# 
# * `clarity (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))` Diamonds can have internal characteristics known as inclusions or external characteristics known as blemishes. Diamonds without inclusions or blemishes are rare; however, most characteristics can only be seen with magnification.  
# 
# `Dimensions`
# 
# `x length in mm (0--10.74)`
# 
# `y width in mm (0--58.9)`
# 
# `z depth in mm (0--31.8)`

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


## Data Ingestion
df = pd.read_csv('data/gemstone.csv')
df.head()


# In[4]:


df.shape


# ### Data Preprocessing

# Steps involved in Data Preprocessing
# 
# - Data cleaning
# - Identifying and removing outliers
# - Encoding categorical variables

# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


## No missing values present in data


# In[8]:


## Drop the id column
df = df.drop(labels=['id'], axis=1)
df.head()


# In[9]:


df.describe()


# In[10]:


## Duplicate records
df.duplicated().sum()


# `Points to notice:`
# 
# Min value of "x", "y", "z" are zero this indicates that there are faulty values in data that represents dimensionless or 2-dimensional diamonds. So we need to filter out those as it clearly faulty data points.

# In[11]:


## Dropping dimensionless diamonds
data = df.drop(df[df["x"]==0].index)
data = df.drop(df[df["y"]==0].index)
data = df.drop(df[df["z"]==0].index)
data.shape


# We lost 10 data points by deleting the dimensionless(2-D or 1-D) diamonds.

# In[12]:


sns.pairplot(data)


# A few points to notice in these pair plots
# 
# There are some features with datapoint that are far from the rest of the dataset which will affect the outcome of our regression model.
# 
# * "y" and "z" have some dimensional outlies in our dataset that needs to be eliminated.
# * The "depth" should be capped but we must examine the regression line to be sure.
# * The "table" featured should be capped too.
# 
# Let's have a look at regression plots to get a close look at the outliers.

# In[13]:


plot_list = ['x', 'y', 'z', 'depth', 'table']

for i in plot_list:
        sns.regplot(x='price', y=i, data=data, color='red', line_kws={'linestyle':'--'}, scatter_kws={'s':50})
        plt.title('Regression Plot of Price and {}'.format(i))
        plt.xlabel('Price')
        plt.ylabel(i)
        plt.show()


# We can clearly spot outliers in these attributes. Next up, we will remove these data points.

# In[14]:


## Dropping the outliers
data = data[(data["depth"]<70.0)&(data["depth"]>54.0)]
data = data[(data["table"]<73)&(data["table"]>50)]
data = data[(data["x"]>2)]
data = data[(data["y"]<9)]
data = data[(data["z"]<6)&(data["z"]>2)]
data.shape


# Now that we have removed regression outliers, let us have a look at the pair plot of data in our hand.

# In[15]:


sns.pairplot(data)


# That's a much cleaner dataset. Next, we will deal with the categorical variables.

# In[16]:


## Segregate numerical and categorical columns

categorical_columns = data.columns[df.dtypes=='object']
numerical_columns = data.columns[df.dtypes!='object']

print('Categorical columns: ', categorical_columns)
print('Numerical columns: ', numerical_columns)


# We have three categorical variables. Let us have a look at them.

# In[17]:


data[categorical_columns].describe()


# In[18]:


data[numerical_columns].describe()


# In[19]:


data.cut.value_counts()


# In[20]:


data.color.value_counts()


# In[21]:


data.clarity.value_counts()


# ### Data Visualization

# Violin Plots for categorical columns

# In[22]:


plt.figure(figsize = (8, 6))

x=0
for i in categorical_columns:
   sns.violinplot(data=data, x=i, y='price')
   plt.title('Violin Plot of {} vs price'.format(i))
   plt.xlabel(i)
   plt.ylabel('Price')
   print('\n')
   plt.show()


# In[23]:


x=0
for i in numerical_columns:
    sns.histplot(data = data, x = i, kde=True)
    print('\n')
    plt.show()


# In[24]:


plt.figure(figsize = (8, 6))

x=0
for i in categorical_columns:
    sns.histplot(data=data, x=i, kde=True)
    print('\n')
    plt.show()


# *Label encoding the data to get rid of object dtype.*

# In[25]:


# Make copy to avoid changing original data 
label_data = data.copy()

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()

for col in categorical_columns:
    label_data[col] = label_encoder.fit_transform(label_data[col])

label_data.head()


# In[26]:


data.describe()


# ### Correlation Matrix

# In[27]:


#correlation matrix
cmap = sns.diverging_palette(70,20,s=50, l=40, n=6,as_cmap=True)
corrmat= label_data.corr()
f, ax = plt.subplots(figsize=(12,12))
sns.heatmap(corrmat,cmap=cmap,annot=True, )


# Points to notice:
# 
# - "x", "y" and "z" show a high correlation to the target column.
# - "depth", "cut" and "table" show low correlation. We could consider dropping but let's keep it.

# ### Model Building

# Steps involved in Model Building
# 
# - Setting up features and target
# - Build a pipeline of standard scalar and model for five different regressors.
# - Fit all the models on training data
# - Get mean of cross-validation on the training set for all the models for negative root mean square error
# - Pick the model with the best cross-validation score
# - Fit the best model on the training set

# In[28]:


label_data.head()


# In[29]:


# Assigning the featurs as X and trarget as y

X= label_data.drop(["price"],axis =1)
y= label_data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=7)


# In[30]:


# Building pipelins of standard scaler and model for varios regressors.

pipeline_lr=Pipeline([("scalar1",StandardScaler()),
                     ("lr_regressor",LinearRegression())])

pipeline_dt=Pipeline([("scalar2",StandardScaler()),
                     ("dt_regressor",DecisionTreeRegressor())])

pipeline_rf=Pipeline([("scalar3",StandardScaler()),
                     ("rf_regressor",RandomForestRegressor())])


pipeline_kn=Pipeline([("scalar4",StandardScaler()),
                     ("kn_regressor",KNeighborsRegressor())])


pipeline_xgb=Pipeline([("scalar5",StandardScaler()),
                     ("xgb_regressor",XGBRegressor())])

# List of all the pipelines
pipelines = [pipeline_lr, pipeline_dt, pipeline_rf, pipeline_kn, pipeline_xgb]

# Dictionary of pipelines and model types for ease of reference
pipe_dict = {0: "LinearRegression", 1: "DecisionTree", 2: "RandomForest",3: "KNeighbors", 4: "XGBoost"}

# Fit the pipelines
for pipe in pipelines:
    pipe.fit(X_train, y_train)


# In[31]:


cv_results_rms = []
for i, model in enumerate(pipelines):
    cv_score = cross_val_score(model, X_train,y_train,scoring="neg_root_mean_squared_error", cv=10)
    cv_results_rms.append(cv_score)
    print("%s: %f " % (pipe_dict[i], cv_score.mean()))


# **Testing the Model with the best score on the test set**
# 
# In the above scores, XGBClassifier appears to be the model with the best scoring on negative root mean square error. Let's test this model on a test set and evaluate it with different parameters.

# ## Model Prediction

# In[32]:


# Model prediction on test data
pred = pipeline_xgb.predict(X_test)


# ## Model Evaluation

# In[33]:


# Model Evaluation
print("R^2:",metrics.r2_score(y_test, pred))
print("Adjusted R^2:",1 - (1-metrics.r2_score(y_test, pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print("MAE:",metrics.mean_absolute_error(y_test, pred))
print("MSE:",metrics.mean_squared_error(y_test, pred))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, pred)))

