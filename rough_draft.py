# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 21:06:35 2015

@author: Danny
"""

# imports
import pandas as pd
import numpy as np

import seaborn as sns

# read in the CSV file from a URL
ncaab = pd.read_csv('https://raw.githubusercontent.com/dansnacks/SF_DAT_15_WORK/master/moredata.csv')
ncaab.dropna()

measureables = ['Position','Height', 'Weight', 'PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TPG', 'FG%', 'FT%', '3P%']

ncaab[['Player', 'BPG', 'Drafted']][ncaab.Position=='C'].sort_index(by='Drafted')

# explore data numerically, looking for differences between species
ncaab.describe()

ncaab.groupby('Position').Height.mean().sort_index(by='Height')

#Percent drafted by position
ncaab.groupby('Position').Drafted.mean()

#Average height by position
ncaab.groupby('Position', as_index=False).Height.mean().sort_index(by='Height', ascending = False)

#Average height by position for drafted players
ncaab[['Height', 'Position']][ncaab['Drafted'] == 1].groupby('Position', as_index=False).mean().sort_index(by='Height', ascending = False)

#Average stats by position
ncaab.groupby('Position', as_index=False).mean().sort_index(by='Height', ascending = False)

#Max stats by position for drafted players
ncaab[ncaab.Drafted == 1].groupby('Position').max()

ncaab[ncaab.Drafted== 0]['PPG'].max()

ncaab[ncaab.Drafted== 1]['PPG'].max()

ncaab[['Height', 'APG'][ncaab.Drafted == 0].groupby('Position').max()

ncaab[[measureables]][ncaab.Drafted == 0].groupby('Position').max()



ncaab[['Height', 'Weight', 'Position', 'BPG', 'RPG']][ncaab['Drafted'] == 1].groupby('Position', as_index=False).mean().sort_index(by='Height', ascending = False)


ncaab.groupby('Position', as_index=False).Height.max().sort_index(by='Height', ascending = False)


ncaab.groupby('Position', 'Drafted').Height.mean().sort_index(by='Height')

ncaab[[ncaab['Drafted'] == 1]].groupby('Position', as_index=False).mean()



# Plot histograms of height by position, 
# remember to share x and share y axis scales!
ncaab.Height.hist(by=ncaab.Position, sharex=True, sharey=True)

def color_drafted(yesno):
    if yesno == 'Y':
        return 'r'
    else:
        return 'k'

colors = ncaab.Drafted.apply(color_drafted)
colors
#see if pattern for height/weight and draft status
ncaab.plot(x='Height', y='Weight', kind='scatter', alpha=0.3, c=colors)

def color_position(position):
    if position == 'C':
        return 'r'
    elif position == 'PF' or position == 'SF' or position == 'F':        
        return 'g'    
    elif position == 'PG' or position == 'SG' or position == 'G':
        return 'b'
    else:
        return 'y'

colorposition = ncaab.Position.apply(color_position)
ncaab.plot(x='Height', y='Weight', kind='scatter', alpha=0.3, c=colorposition)



fig, axs = plt.subplots(1, 3, sharey=True)
ncaab.plot(kind='scatter', x='Height', y='Weight', ax=axs[0], figsize=(16, 6))
ncaab.plot(kind='scatter', x='RPG', y='Weight', ax=axs[1], figsize=(16, 6))
ncaab.plot(kind='scatter', x='FG%', y='Weight', ax=axs[2], figsize=(16, 6))

#scatter plot and regression vs Weight
sns.pairplot(ncaab, x_vars=['RPG','BPG','FG%'], y_vars='Weight', size=4.5, aspect=0.7, kind='reg')
sns.pairplot(ncaab, x_vars=['APG','SPG','FT%'], y_vars='Weight', size=4.5, aspect=0.7, kind='reg')


sns.pairplot(ncaab, x_vars=['Height','RPG','FG%'], y_vars='Weight', size=4.5, aspect=0.7, kind='reg')


#heatmap for Drafted Y/N - next three lines
df=pd.DataFrame(ncaab)
g = sns.FacetGrid(df, col='Drafted')
g.map_dataframe(lambda data, color: sns.heatmap(data.corr(), linewidths=0))

df=pd.DataFrame(ncaab)
g = sns.FacetGrid(df)
g.map_dataframe(lambda data, color: sns.heatmap(data.corr(), linewidths=0))

#Parallel coordinates
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates
features = ['PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TPG', 'FG%', '3P%', 'Position']
ncaab_df = pd.DataFrame(ncaab[ncaab.Drafted=='N'][ncaab.Position!='F'][ncaab.Position!='G'], columns = features)
parallel_coordinates(data=ncaab_df, class_column='Position')

plt.figure()


features = ['Height', 'Weight', 'PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TPG', 'FG%', '3P%', 'Position']
ncaab_df = pd.DataFrame(ncaab[ncaab.Drafted== 1], columns = features)
parallel_coordinates(data=ncaab_df, class_column='Position')

plt.figure()

#heat map for relevant columns
df=pd.DataFrame(ncaab, columns = features)
g = sns.FacetGrid(df)
g.map_dataframe(lambda data, color: sns.heatmap(data.corr(), linewidths=.1))

df=pd.DataFrame(ncaab[ncaab['Drafted'] == 1], columns = features)
g = sns.FacetGrid(df)
g.map_dataframe(lambda data, color: sns.heatmap(data.corr(), linewidths=.1))

df=pd.DataFrame(ncaab[ncaab['Drafted'] == 0], columns = features)
g = sns.FacetGrid(df)
g.map_dataframe(lambda data, color: sns.heatmap(data.corr(), linewidths=.1))


###standardize axes
ncaab['stdPPG'] = (ncaab.PPG - np.mean(ncaab.PPG) ) / np.std(ncaab.PPG) 
ncaab['stdRPG'] = (ncaab.RPG - np.mean(ncaab.RPG) ) / np.std(ncaab.RPG) 
ncaab['stdAPG'] = (ncaab.APG - np.mean(ncaab.APG) ) / np.std(ncaab.APG) 
ncaab['stdSPG'] = (ncaab.SPG - np.mean(ncaab.SPG) ) / np.std(ncaab.SPG) 
ncaab['stdBPG'] = (ncaab.BPG - np.mean(ncaab.BPG) ) / np.std(ncaab.BPG) 
ncaab['stdTPG'] = (ncaab.TPG - np.mean(ncaab.TPG) ) / np.std(ncaab.TPG) 

ncaab[ncaab['Position'] == 'C'][ncaab['Drafted'] == 1].mean() - ncaab[ncaab['Position'] == 'C'][ncaab['Drafted'] == 'N'].mean()

#################          Difference Between Drafted and Non-Drafted    #################

#percentage better drafted vs undrafted C
((ncaab[ncaab['Position'] == 'C'][ncaab['Drafted'] == 1].mean()
 - ncaab[ncaab['Position'] == 'C'][ncaab['Drafted'] == 0].mean()) * 100/ 
 ncaab[ncaab['Position'] == 'C'][ncaab['Drafted'] == 1].mean())

#difference between drafted and not drafted PG
features = ['stdPPG', 'stdRPG', 'stdAPG', 'stdSPG', 'stdBPG', 'stdTPG', 'FG%', '3P%', 'Drafted']
ncaab_df = pd.DataFrame(ncaab[ncaab.Position=='PG'], columns = features)
parallel_coordinates(data=ncaab_df, class_column='Drafted')

ncaab[ncaab['Position'] == 'PG'][ncaab['Drafted'] == 1].mean() - ncaab[ncaab['Position'] == 'PG'][ncaab['Drafted'] == 'N'].mean()

#percentage better drafted vs undrafted PG
((ncaab[ncaab['Position'] == 'PG'][ncaab['Drafted'] == 1].mean()
 - ncaab[ncaab['Position'] == 'PG'][ncaab['Drafted'] == 0].mean()) * 100/ 
 ncaab[ncaab['Position'] == 'PG'][ncaab['Drafted'] == 1].mean())

 #difference between drafted and not drafted SG
features = ['stdPPG', 'stdRPG', 'stdAPG', 'stdSPG', 'stdBPG', 'stdTPG', 'FG%', '3P%', 'Drafted']
ncaab_df = pd.DataFrame(ncaab[ncaab.Position=='SG'], columns = features)
parallel_coordinates(data=ncaab_df, class_column='Drafted')

ncaab[ncaab['Position'] == 'SG'][ncaab['Drafted'] == 1].mean() - ncaab[ncaab['Position'] == 'SG'][ncaab['Drafted'] == 'N'].mean()

#percentage better drafted vs undrafted SG
((ncaab[ncaab['Position'] == 'SG'][ncaab['Drafted'] == 1].mean()
 - ncaab[ncaab['Position'] == 'SG'][ncaab['Drafted'] == 0].mean()) * 100/ 
 ncaab[ncaab['Position'] == 'SG'][ncaab['Drafted'] == 1].mean())
#Shows HUGE differences between drafted and undrafted players

ncaab[ncaab['Position'] == 'C'][ncaab['Drafted'] == 1].Player.count() 
ncaab[ncaab['Position'] == 'PG'][ncaab['Drafted'] == 1].Player.count() 
ncaab[ncaab['Position'] == 'SG'][ncaab['Drafted'] == 1].Player.count() 
ncaab[ncaab['Position'] == 'PF'][ncaab['Drafted'] == 1].Player.count() 
ncaab[ncaab['Position'] == 'SF'][ncaab['Drafted'] == 1].Player.count() 
ncaab[ncaab['Position'] == 'G'][ncaab['Drafted'] == 1].Player.count()
ncaab[ncaab['Position'] == 'F'][ncaab['Drafted'] == 1].Player.count()


# fit a linear regression model and store the class predictions

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
feature_cols = ['Height', 'Weight', 'PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TPG', 'FG%', 'FT%', '3P%']
X = ncaab[ncaab['Position'] == 'C'][feature_cols]
y = ncaab[ncaab['Position'] == 'C']['Drafted']
logreg.fit(X, y)
assorted_pred_class = logreg.predict(X)

assorted_pred_prob = logreg.predict_proba(X)[:, 1]

#########################              Logistic Regression              #######################
from sklearn import metrics

# model for PG
feature_cols = ['PPG', 'APG', 'BPG']
X = ncaab[ncaab['Position'] == 'PG'][feature_cols]
y = ncaab[ncaab['Position'] == 'PG']['Drafted']

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

preds = logreg.predict(X_test)
print metrics.confusion_matrix(y_test, preds)

#model for SG
feature_cols = ['PPG', 'RPG', 'APG', 'TPG']
X = ncaab[ncaab['Position'] == 'SG'][feature_cols]
y = ncaab[ncaab['Position'] == 'SG']['Drafted']

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

preds = logreg.predict(X_test)
print metrics.confusion_matrix(y_test, preds)

#model for PF
feature_cols = ['PPG', 'RPG', 'SPG', 'TPG', '3P%']
X = ncaab[ncaab['Position'] == 'PF'][feature_cols]
y = ncaab[ncaab['Position'] == 'PF']['Drafted']

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

preds = logreg.predict(X_test)
print metrics.confusion_matrix(y_test, preds)

#model for SF
feature_cols = ['PPG', 'APG', 'SPG', 'TPG']
X = ncaab[ncaab['Position'] == 'SF'][feature_cols]
y = ncaab[ncaab['Position'] == 'SF']['Drafted']

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

preds = logreg.predict(X_test)
print metrics.confusion_matrix(y_test, preds)

#model for C
feature_cols = ['PPG', 'Rank', 'RPG', 'APG', 'SPG', 'BPG']
X = ncaab[ncaab['Position'] == 'C'][feature_cols]
y = ncaab[ncaab['Position'] == 'C']['Drafted']

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

preds = logreg.predict(X_test)
print metrics.confusion_matrix(y_test, preds)


#overall model + confusion matrix
feature_cols = ['PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TPG', 'FG%', 'FT%', '3P%']
X = ncaab[feature_cols]
y = ncaab['Drafted']

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

preds = logreg.predict(X_test)
print metrics.confusion_matrix(y_test, preds)


########################              Decision Tree             ########################

from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

ncaabtree = ncaab[['Rank', 'MIN', 'PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TPG', 'FG%', 'FT%', '3P%', 'Position', 'Drafted']]

ncaabtree['Position'] = np.where(ncaabtree.Position == 'G', 1, ncaabtree.Position)
ncaabtree['Position'] = np.where(ncaabtree.Position == 'F', 2, ncaabtree.Position)
ncaabtree['Position'] = np.where(ncaabtree.Position == 'PG', 3, ncaabtree.Position)
ncaabtree['Position'] = np.where(ncaabtree.Position == 'SG', 4, ncaabtree.Position)
ncaabtree['Position'] = np.where(ncaabtree.Position == 'PF', 5, ncaabtree.Position)
ncaabtree['Position'] = np.where(ncaabtree.Position == 'SF', 6, ncaabtree.Position)
ncaabtree['Position'] = np.where(ncaabtree.Position == 'C', 7, ncaabtree.Position)

Drafted = ncaabtree['Drafted']
del ncaabtree['Drafted']

X_train, X_test, y_train, y_test = train_test_split(ncaabtree,Drafted, random_state=1)

datatree = tree.DecisionTreeClassifier(random_state=1, max_depth=3)

datatree.fit(X_train, y_train)

treefeatures = X_train.columns.tolist()

datatree.classes_

datatree.feature_importances_

pd.DataFrame(zip(treefeatures, datatree.feature_importances_)).sort_index(by=1, ascending=False)

preds = datatree.predict(X_test)

metrics.accuracy_score(y_test, preds)

pd.crosstab(y_test, preds, rownames=['actual'], colnames=['predicted'])

probs = datatree.predict_proba(X_test)[:,1]

metrics.roc_auc_score(y_test, probs)

'''

FINE-TUNING THE TREE

'''
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV



# Conduct a grid search for the best tree depth
datatree = tree.DecisionTreeClassifier(random_state=1)
depth_range = range(1, 8)
param_grid = dict(max_depth=depth_range)
grid = GridSearchCV(datatree, param_grid, cv=5, scoring='roc_auc')
grid.fit(ncaabtree, Drafted)

# Check out the scores of the grid search
grid_mean_scores = [result[1] for result in grid.grid_scores_]

# Plot the results of the grid search
plt.figure()
plt.plot(depth_range, grid_mean_scores)
plt.hold(True)
plt.grid(True)
plt.plot(grid.best_params_['max_depth'], grid.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')
         
#best tree depth model
datatree = tree.DecisionTreeClassifier(max_depth=3)
np.mean(cross_val_score(datatree, ncaabtree, Drafted, cv=5, scoring='roc_auc'))


best = grid.best_estimator_

cross_val_score(best, ncaabtree, Drafted, cv=10, scoring='roc_auc').mean()

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
datatree = tree.DecisionTreeClassifier(random_state=1, max_depth=2)

cross_val_score(logreg, ncaabtree, Drafted, cv=10, scoring='roc_auc').mean()


