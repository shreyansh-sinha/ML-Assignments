import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
# get_ipython().magic(u'matplotlib inline')

from sklearn.tree import DecisionTreeClassifier

from sklearn.decomposition import PCA

from mlxtend.feature_selection import SequentialFeatureSelector as SFS


# ### Loading the data

# In[65]:


column_names = ["erythema", "scaling", "definite borders", "itching",
                "koebner phenomenon", "polygonal papules", "follicular papules", "oral mucosal involvement", 
               "knee and elbow involvement", "scalp involvement", "family history", "melanin incontinence", "eosinophils in the infiltrate", "PNL infiltrate", "fibrosis of the papillary dermis", "exocytosis", 
               "acanthosis", "hyperkeratosis", "parakeratosis", "clubbing of the rete ridges", "elongation of the rete ridges", "thinning of the suprapapillary epidermis", "spongiform pustule", "munro microabcess", 
               "focal hypergranulosis", "disappearance of the granular layer", "vacuolisation and damage of basal layer", "spongiosis", "saw-tooth appearance of retes", "follicular horn plug", "perifollicular parakeratosis", "inflammatory monoluclear inflitrate",
               "band-like infiltrate", "age", "type"]

data = pd.read_csv("./datasets/dermatology.data", header = None, names = column_names)
data = data[data['age'] != '?']

for i in data.columns:
    data[i].apply(int)
    
X = data.drop("type", 1)
y = data["type"]


# In[66]:

# Pandas dataframe.corr()
# is used to find the pairwise correlation of all columns in the dataframe.
corr = data.corr()

plt.figure(figsize=(12,12))
sns.heatmap(corr, linewidths=.5, cmap='viridis')


# ### Forward selection

# In forward selection we start with NULL MODEL and then start fitting 
# the model with each individual feature one at a time and select the 
# feature with min p-value. Now fit the model with two features by 
# trying combinations of eariler selected feature with all other remaining
# features and then same goes for three.........
# this process stops when we have a set of selected features with p-value of 
# individual feature less than significance level

# In[67]:


sfs = SFS(DecisionTreeClassifier(),
           k_features=10,
           forward=True,
           floating=False,
           scoring = None,
           cv = 0)

sfs.fit(X, y)
sfs.k_feature_names_


# Backward Elimination

'''
In backward elimination, we start with the full model 
(including all the independent variables) and then 
remove the insignificant feature with highest p-value(> significance level). 
his process repeats again and again until we have the final set of 
significant features.

'''

sfs = SFS(DecisionTreeClassifier(),
           k_features=10,
           forward=False,
           floating=False,
           scoring = None,
           cv = 0)

sfs.fit(X, y)
sfs.k_feature_names_


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[51]:

# feature scaling
# fitting the standard scale (mean = 0, variance = 1)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)


# In[78]:

# pca is a procedure that uses an orthogonal transformation to convert
# a set of oservations of possibly correlated variables, into a set of 
# values of linearly uncorrelated variables.

# Transformation is defined under the constraint that first principal component
# has largest possible variance and each succeeding component in turn has
# the highest variance under the constraint that it is orthogonal to the 
# preceding components

pca = PCA()
pca.fit_transform(scaled_data)

# tells the pca components ==>
pca.components_

# explained variance tells how much info(variance) can be attributed to
# principal components. This is important cuz we reduce the numbber of dimensions

'''

The fraction of variance explained by a principal component 
is the ratio between the variance of that principal component 
and the total variance. For several principal components, add up 
their variances and divide by the total variance

'''
explained_variance=pca.explained_variance_ratio_
print(explained_variance)

# In[77]:


plt.figure(figsize=(8, 6))

plt.bar(range(34), explained_variance, alpha=0.5, align='center',
        label='Explained Variance representation')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()


# In[54]:

# information gain

from sklearn.feature_selection import f_classif


# In[55]:


f, p = f_classif(X, y)


clf = DecisionTreeClassifier()


# In[59]:


clf.fit(X, y)


# In[85]:


clf.feature_importances_


# In[86]:


column_dict = {}
for name, val in zip(X.columns, clf.feature_importances_):
    column_dict[val] = name


# In[87]:


sort_dict = [(column_dict[cval], cval) for cval in sorted(column_dict, reverse = True)]


# In[88]:


sort_dict


# In[89]:

# based on variance 

from sklearn.feature_selection import mutual_info_classif



"""
Returns a value between 0 to 1. Higher the value, the more the dependence
The function relies on nonparametric methods based on entropy estimation 
from k-nearest neighbors distances
"""

mutual_info_classif(X, y)


column_dict = {}
for val, name in zip(mutual_info_classif(X, y), column_names):
    column_dict[val] = name
    
sort_dict = [(column_dict[cval], cval) for cval in sorted(column_dict, reverse = True)]

sort_dict


X.var(axis = 0).sort_values(ascending = False)[0:11]
