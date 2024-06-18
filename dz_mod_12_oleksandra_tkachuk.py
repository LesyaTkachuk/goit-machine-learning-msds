# %%
import warnings
import pickle
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
from kneed import KneeLocator
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# %%
# load data
with open('./datasets/mod_05_topic_10_various_data.pkl', 'rb') as fl:
    datasets = pickle.load(fl)

data=datasets['concrete']
data.head()

# %%
# get info about feature types
data.info()

# %%
# get dataset statistics
data.describe()

# %%
components=['Cement',
              'BlastFurnaceSlag',
              'FlyAsh',
              'Water',
              'Superplasticizer',
              'CoarseAggregate',
              'FineAggregate']
data['Count']=data[components].gt(0).sum(axis=1)
data.head()

# %%
# data normalization
X = StandardScaler().fit_transform(data)

pca = PCA(random_state=42).fit(X)
pve = pca.explained_variance_ratio_


# %%
# visualize kneed for the given dataset
sns.set_theme()

kneedle = KneeLocator(x=range(1, len(pve)+1), y=pve, curve='convex', direction='decreasing')

kneedle.plot_knee()

plt.title(f'Knee Point at {kneedle.elbow+1}')
plt.show()

# %%
n_components = kneedle.elbow

ax = sns.lineplot(np.cumsum(pve))
ax.axvline(x=n_components, c='black',
           linestyle='--',
           linewidth=0.75)

ax.axhline(y=np.cumsum(pve)[n_components], c='black',
           linestyle='--',
           linewidth=0.75)

ax.set(xlabel='number of components', ylabel='cummulative explained vatiance')


# %%

X = pca.transform(X)[:, :n_components]

# define optimal number of clusters
model_kmn_vis = KMeans(random_state=42)

visualizer = KElbowVisualizer(model_kmn_vis, k=(2,10), timing=False)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    
    visualizer.fit(X)
    
visualizer.show()

# %%
# create clustering model
k_best = visualizer.elbow_value_

model_kmn = KMeans(n_clusters=k_best, random_state=42).fit(X)
labels_kmn=pd.Series(model_kmn.labels_, name='k-means')

labels_kmn

# %%
data['Cluster']=model_kmn.labels_
data.head()

# %%
# Clusters visualisation
fig,ax= plt.subplots( figsize=(10, 5))

sns.scatterplot(x=X[:, 0],
                    y=X[:, 1],
                    hue=labels_kmn,
                    style=labels_kmn,
                    edgecolor='black',
                    linewidth=0.5,
                    s=60,
                    palette='tab20',
                    legend=False,
                    ax=ax)

ax.set(title=labels_kmn.name)


# %%
# make report dataset
report=(data.groupby('Cluster')[components]).mean()
report['Components']=(data.groupby('Cluster')['Count']).mean()
report['Count']=(data.groupby('Cluster')['Cluster']).count()

# %%
# Conclusion
# With the help of elbow method from kneedle library we defined the optimal number of clusters as 5 and performed clusterisation with the help of KMeans model.
# We created report dataframe, which depicts mean values for all components, number of components in one object and total number of objects that belong to each class
# We can observe that clustering could be refered according to the mean value of each component in the set of objects. For example, we see that cluster 4 contains objects that don't have 'FlyAsh' and 'Superplasticizer'
# and cluster 2 contains only smallest amount of the apointed components. Cluster 0 contains the highest amount of 'Cement' and 'Superplasticizer'.
# We can make a conclusion that our model is quit good while solving clustering based on the components amount in the given object (receipt).
