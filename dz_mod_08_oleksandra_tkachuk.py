# %%
import warnings
import pandas as pd
import numpy as np
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# load train and validation datasets
train_data=pd.read_csv('./datasets/mod_04_hw_train_data.csv')
valiid_data=pd.read_csv('./datasets/mod_04_hw_valid_data.csv')

# %%
# observe data 
train_data.head()

# %%
# observe data types and missed data
train_data.info()

# %%
# get dataset statistics
train_data.describe()

# %%
#  primary feature investigation
imp_features=['Experience', 'Qualification', 'University','Role', 'Cert']
X_train=train_data[imp_features]
y_train=train_data['Salary']

X_valid=valiid_data[imp_features]
y_valid=valiid_data['Salary']