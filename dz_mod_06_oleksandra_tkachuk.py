import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv('./datasets/mod_03_topic_05_weather_data.csv.gz')
data.shape

data.head()

# get info about feature types
data.dtypes

# check for missed values
data.isna().mean().sort_values(ascending=False)

# drop rows in which missed values reach more than 35%
data=data[data.columns[data.isna().mean().lt(0.35)]]

# drop rows where target is missed
data=data.dropna(subset="RainTomorrow")

data.shape

# split data on numerical and categorical subsets
data_num=data.select_dtypes(include=np.number)
data_cat=data.select_dtypes(include='object')

# convert Date column to datetime type
data_cat["Date"]=pd.to_datetime(data['Date'])

# create two new columns 'Year' and 'Month'
data_cat[['Year', 'Month']]=(data_cat["Date"].apply(lambda x: pd.Series([x.year, x.month])))

# move 'Year' column to numerical features subset
data_num['Year']=data_cat['Year']

data_cat.drop(['Date','Year'], axis=1, inplace=True)

# check subsets
data_cat['Month'].head()
data_num["Year"].head()

# split on train and test datasets (test datasets include data for the last year, train dataset - the rest)
X_test_num=data_num[-365:]
X_test_cat=data_cat[-365:]

X_train_num=data_num[:-365]
X_train_cat=data_cat[:-365]

y_train=X_train_cat['RainTomorrow']
y_test=X_test_cat['RainTomorrow']

X_test_cat=X_test_cat.drop('RainTomorrow',axis=1)
X_train_cat=X_train_cat.drop('RainTomorrow',axis=1)

# restore missed data to mean value for numerical data
num_imputer=SimpleImputer().set_output(transform='pandas')
X_train_num=num_imputer.fit_transform(X_train_num)
X_test_num=num_imputer.transform(X_test_num)

pd.concat([X_train_num, X_test_num]).isna().sum()

# restore missed data to mode value for categorical data
cat_imputer=SimpleImputer(strategy='most_frequent').set_output(transform='pandas')
X_train_cat=cat_imputer.fit_transform(X_train_cat)
X_test_cat=cat_imputer.transform(X_test_cat)

pd.concat([X_train_cat, X_test_cat]).isna().sum()

# perform numeric features normalization
scaler=StandardScaler().set_output(transform='pandas')
X_train_num=scaler.fit_transform(X_train_num)
X_test_num=scaler.transform(X_test_num)

# encode categorical data
encoder=(OneHotEncoder(drop='if_binary',sparse_output=False).set_output(transform='pandas'))
X_train_cat=encoder.fit_transform(X_train_cat)
X_test_cat=encoder.transform(X_test_cat)

X_train_cat.shape

# concat categorical and numeric subsets for model creation
X_train=pd.concat([X_train_num, X_train_cat], axis=1)
X_test=pd.concat([X_test_num, X_test_cat], axis=1)

X_train.shape

# check target value distribution
y_train.value_counts(normalize=True)


# build LogisticRegression training model with parameters solver='liblinear' and class_weight='balanced' to encount disbalance of the target value
clf=(LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42).fit(X_train, y_train))

pred=clf.predict(X_test)


# build confusion matrix to observe results
ConfusionMatrixDisplay.from_predictions(y_test, pred)

plt.show()

# get accuracy metrics using classification_report method
print(classification_report(y_test, pred))




# build LogisticRegression training model with parameters solver='newton-cholesky' and class_weight='balanced' to encount disbalance of the target value
clf=(LogisticRegression(solver='newton-cholesky', class_weight='balanced', random_state=42).fit(X_train, y_train))

pred=clf.predict(X_test)


# build confusion matrix to observe results
ConfusionMatrixDisplay.from_predictions(y_test, pred)

plt.show()

# get accuracy metrics using classification_report method
print(classification_report(y_test, pred))

# Conclusion, we have build LogisticRegression model for Rain prediction and obtained results with high accuracy 93%, as well as all other accuracy metrics that are in range from 95%-97%. These results are much better as results in our practical lesson model. We can make a conclusion that splitting our dataset on train and test subsets using last year as conditional "future" period allows us to improve our model prediction significantly in comparison with just usual train_test_split method.  





































