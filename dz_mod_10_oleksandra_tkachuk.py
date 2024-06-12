# %%
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn import tree

# %%
with open('./datasets/mod_05_topic_10_various_data.pkl', 'rb') as fl:
    datasets = pickle.load(fl)

# %%
# load autos dataset
autos = datasets['autos']
autos.head()

# %%
# observe data types and missed data
autos.info()

# %%
# get dataset statistics
autos.describe()

# %%
# get list of descrete features
X=autos.copy()
y=X.pop('price')

cat_features=X.select_dtypes('object').columns
cat_features

# %%
# convert descrete features into numerical ones
for colname in cat_features:
    X[colname], _ = X[colname].factorize()
    
X.dtypes

# %%
# define mutual information scores
mi_scores = mutual_info_regression(X, y, discrete_features=X.columns.isin(cat_features.to_list()+['num_of_doors', 'num_of_cylinders']), random_state=42)

mi_scores = (pd.Series(mi_scores, name="MI Scores", index=X.columns).sort_values())

mi_scores

# %%
# visualize the results
plt.figure(figsize=(10,10), dpi=196)
plt.barh(np.arange(len(mi_scores)), mi_scores)
plt.yticks(np.arange(len(mi_scores)), mi_scores.index)
plt.title('Mutual Information Scores')

# plt.savefig('./mi_scores.png')
plt.show()

# %%
# split dataset on test and train subsets
X_train, X_test, y_train, y_test = (train_test_split(autos.drop('price', axis=1), autos['price'], test_size=0.3, random_state=42))

# %%
# encode categorial features
encoder = ce.TargetEncoder()
X_train = encoder.fit_transform(X_train, y_train)
X_test = encoder.transform(X_test)


# %%
# build RandomForestRegression model
mod_rnd_frs = (RandomForestRegressor(random_state=42).fit(X_train, y_train))

prd_rnd_frs = mod_rnd_frs.predict(X_test)

mape = mean_absolute_percentage_error(y_test, prd_rnd_frs)

print(f'Accuracy: {1 - mape:.1%}')

# %%
# build RandomForestRegression model with max depth equals to 4 
mod_rnd_frs_upd = (RandomForestRegressor(random_state=42, max_depth=7).fit(X_train, y_train))

prd_rnd_frs_upd = mod_rnd_frs_upd.predict(X_test)

mape_upd = mean_absolute_percentage_error(y_test, prd_rnd_frs_upd)

print(f'Accuracy of updated decision tree: {1 - mape_upd:.1%}')

# %%
# define feature importance in decision tree model
plt.figure(figsize=(10,10), dpi=196)
feat_importance_scores=pd.Series(data=mod_rnd_frs_upd.feature_importances_, index=X.columns).sort_values(ascending=True)
plt.barh(np.arange(len(feat_importance_scores)), feat_importance_scores)
plt.yticks(np.arange(len(feat_importance_scores)), feat_importance_scores.index)
plt.title('Feature Importance Scores')
#(pd.Series(data=mod_rnd_frs.feature_importances_, index=X.columns).sort_values(ascending=True).plot.barh())

# plt.savefig('./dec_tree_features_importance.png')
plt.show()

# %%
# rank mutual information scores
frame_mi={'feature_mi': mi_scores.index, 'MI Score': mi_scores}
mi_scores_df=pd.DataFrame(frame_mi)

mi_scores_df['pct_rank_mi']=mi_scores_df['MI Score'].rank(pct=True)
mi_scores_df

# %%
# rank feature importance scores
frame_fi={'feature_fi': feat_importance_scores.index, 'FI Score': feat_importance_scores}
fi_scores_df=pd.DataFrame(frame_fi)

fi_scores_df['pct_rank_fi']=fi_scores_df['FI Score'].rank(pct=True)
fi_scores_df

# %%
# visualize grouped barsplots for mutual information and feature importance scores comparison
scores_df=pd.concat([mi_scores_df, fi_scores_df],axis=1, join="inner")

scores_df.drop('feature_fi', axis=1, inplace=True)


scores_df=scores_df.rename(columns={'feature_mi':'feature'})


# %%
melted=pd.melt(scores_df, id_vars=['feature'], value_vars=['pct_rank_mi', 'pct_rank_fi'])

sns.set_theme(style="whitegrid")
sns.catplot(data=melted, kind='bar', x='value', y='feature', hue='variable', errorbar="sd", palette="dark", alpha=.6, height=6)

# plt.savefig('./grouped_rank.png')
plt.show()

# %%
# Conclusion
# Observing grouping barsplots we can see that not all features have the same importance comparing different scores such as mutual information score and feature importance score.
# We see that 'curb_weight' and 'engine_size' are the most important features in the dataset according to both scores.
# But for example such features as 'make', 'height', 'engine_type' have lower importance according to mutual information score, but have great importance in Decision Tree model according to it's characteristics ('Gini Impurity', 'Entropy', 'Misclassification Error') 
# And such feature as 'highway_mpg', 'length', 'wheel_base' vise versa have lower importance for the given model. 
# So we can conclude that we should investigate feature importance for each model and use different methods and technics based on model type, different scores comparison, domain specific and etc.

# %%
# build RandomForestRegression model without unimportant features according to .feature_importances_() method
unimp_features=['num_of_doors', 'engine_location', 'compression_ratio', 'wheel_base', 'num_of_cylinders']
mod_rnd_frs_upd = (RandomForestRegressor(random_state=42).fit(X_train.drop(unimp_features, axis=1), y_train))

prd_rnd_frs_upd = mod_rnd_frs_upd.predict(X_test.drop(unimp_features, axis=1))

mape_upd = mean_absolute_percentage_error(y_test, prd_rnd_frs_upd)

print(f'Accuracy of decision tree without unimportant feutures: {1 - mape_upd:.1%}')

# We see that deleting of features with the lowest importance doesn't lead to higher model accuracy in our case





