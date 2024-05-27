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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# import california housing dataset
california_housing = fetch_california_housing(as_frame=True)
california_housing.data.head()

# define target value
california_housing.target.head()

# check dataset for missing values and get basic info
california_housing.frame.info()

data=california_housing.frame

# build feature histograms to indificate distribution type and abnormal values
sns.set_theme()

melted = data.melt()

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
data[features_of_interest].describe()

# delete abnormal values from dataset
subset=data[features_of_interest]
z_scores=subset.apply(zscore)
z_scores

outliers=(np.abs(z_scores>3)).any(axis=1)
outliers_idx=data[outliers].index
outliers_idx

data_cleared=data.drop(outliers_idx)
data_cleared.shape

# build feature histograms after abnormal values dropping
sns.set_theme()

melted_cleared = data_cleared.melt()

g1 = sns.FacetGrid(melted_cleared,
                  col='variable',
                  col_wrap=3,
                  sharex=False,
                  sharey=False)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    g1.map(sns.histplot, 'value')

g1.set_titles(col_template='{col_name}')

g1.tight_layout()

plt.show()

# build visualisation of house price depending on 'Longitude' and 'Latitude' 
fig, ax = plt.subplots(figsize=(6, 5))

sns.scatterplot(
    data=california_housing.frame,
    x='Longitude',
    y='Latitude',
    size='MedHouseVal',
    hue='MedHouseVal',
    palette='viridis',
    alpha=0.5,
    ax=ax)

plt.legend(
    title='MedHouseVal',
    bbox_to_anchor=(1.05, 0.95),
    loc='upper left')

plt.title('Median house value depending of\n their spatial location')

# we see that 'Longitude' and 'Latitude' are important features and couldn't be excluded from a model

# build correlation matrix
columns_drop = ['Longitude', 'Latitude']
corr_mtx = data_cleared.drop(columns=columns_drop).corr()

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

# 'AveRooms' and 'MedInc' features have big positive correlation. But 'MedInc' has higher correlation with our target value, that's why couldn't be excluded
# Let's delete 'AveRooms' feature from our model
data_cleared=data_cleared.drop(columns=['AveRooms'])

data_cleared.head()

# split on train and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    data_cleared.drop(columns=['MedHouseVal']),
    data_cleared['MedHouseVal'],
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

# polynomial regression 
# transform features to polynomial ones and estimate polinomial regression model
poly=PolynomialFeatures(2).set_output(transform='pandas')

Xtr=poly.fit_transform(X_train_scaled)
Xts=poly.transform(X_test_scaled)

model_upd=LinearRegression().fit(Xtr, y_train)
y_pred_upd=model_upd.predict(Xts)
y_pred_upd=pd.Series(y_pred_upd, index=Xts.index).clip(ymin, ymax)

r_sq_upd=model_upd.score(Xtr, y_train)
mae_upd=mean_absolute_error(y_test, y_pred_upd)
mape_upd=mean_absolute_percentage_error(y_test,y_pred_upd)

print(f'R2: {r_sq_upd:.2f} | MAE: {mae_upd:.2f} | MAPE: {mape_upd:.2f}')

# Conclusion
# After clearing our dataset from abnormal values we have reached higher model accuracy in comparison with a model from a practical lesson. We reached 71% accuracy with simple linear regression model and 74% accuracy with polynomial regression model (against 69% and 73% correspondently).
# We can notice that data preparation and cleanning are important steps which result in a model accuracy.