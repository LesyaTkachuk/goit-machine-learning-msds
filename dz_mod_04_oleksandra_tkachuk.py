import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import zscore
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# import california housing dataset
california_housing = fetch_california_housing(as_frame=True)
california_housing.data.head()

# define target value
california_housing.target.head()

# check dataset for missing values and get basic info
california_housing.frame.info()

# build feature histograms to indificate distribution type and abnormal values
sns.set_theme()

melted = california_housing.frame.melt()

g = sns.FacetGrid(melted,
                  col='variable',
                  col_wrap=3,
                  sharex=False,
                  sharey=False)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    g.map(sns.histplot, 'value')

g.set_titles(col_template='{col_name}')

g.tight_layout()

plt.show()

# describe basic features
features_of_interest = ['AveRooms', 'AveBedrms', 'AveOccup', 'Population']
california_housing.frame[features_of_interest].describe()

# delete abnormal values from dataset
df_zscore_averooms = zscore(california_housing.data['AveRooms'], nan_policy='omit')
df_zscore_avebedrms = zscore(california_housing.data['AveBedrms'], nan_policy='omit')
df_zscore_aveoccup = zscore(california_housing.data['AveOccup'], nan_policy='omit')
df_zscore_population = zscore(california_housing.data['Population'], nan_policy='omit')

df_zscore_averooms
df_zscore_averooms.describe()

# get the row where AveRooms=8.288136
#california_housing.data[california_housing.data['AveRooms']==8.288136]

# ??????????????????????????????????????????????????????????????????????

# to get row index
#row_index=california_housing.data.index[california_housing.data['AveRooms']==8.288136].tolist()
#row_index

# substitute values that are greater in 3-sigma times
#california_housing.data['AveRooms']=california_housing.data['AveRooms'].apply(lambda x: x if df_zscore_averooms[raw_index]<3 AND df_zscore_averooms[raw_index]>-3   else california_housing.data['AveRooms'].mean() )



#outliers=np.where(df_zscore_averooms.between(-3,3))
#outliers

california_housing.data['AveRooms'].describe()

# build correlation matrix
corr_mtx = california_housing.frame.corr()

mask_mtx = np.zeros_like(corr_mtx)
np.fill_diagonal(mask_mtx, 1)

fig, ax = plt.subplots(figsize=(7, 6))

sns.heatmap(corr_mtx,
            cmap='coolwarm',
            center=0,
            annot=True,
            fmt='.2f',
            linewidth=0.5,
            square=True,
            mask=mask_mtx,
            ax=ax)

plt.show()

# 'Latitude' and 'Longitude' feature have hight positive correlation. Let's delete 'Latitude' feature from our model
california_housing_new=california_housing.data.drop(columns=['Latitude'])

california_housing_new.head()

# split on train and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    california_housing_new,
    california_housing.target,
    test_size=0.2,
    random_state=42)

# perform data normalisation
scaler = StandardScaler().set_output(transform='pandas').fit(X_train)

X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

X_train_scaled.describe()

# build a model
model = LinearRegression().fit(X_train_scaled, y_train)
y_pred=model.predict(X_test_scaled)
ymin, ymax = y_train.agg(['min', 'max']).values
y_pred=pd.Series(y_pred, index=X_test_scaled.index).clip(ymin, ymax)
y_pred.head()

# model accuracy estimation
r_sq=model.score(X_train_scaled, y_train)
mae=mean_absolute_error(y_test, y_pred)
mape=mean_absolute_percentage_error(y_test, y_pred)

print(f'R2: {r_sq:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}')