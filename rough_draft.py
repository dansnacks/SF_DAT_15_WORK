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
ncaab = pd.read_csv('https://raw.githubusercontent.com/dansnacks/SF_DAT_15_WORK/master/project_data.csv')

ncaab.dropna()

measureables = ['Position','Height', 'Weight', 'PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TPG', 'FG%', 'FT%', '3P%']

ncaab[['Player', 'BPG', 'Drafted']][ncaab.Position=='C'].sort_index(by='Drafted')

# explore data numerically, looking for differences between species
ncaab.describe()

ncaab.groupby('Position').Height.mean().sort_index(by='Height')
iris.groupby('species')['sepal_length', 'sepal_width', 'petal_length', 'petal_width'].mean()
iris.groupby('species').describe()

#Average height by position
ncaab.groupby('Position', as_index=False).Height.mean().sort_index(by='Height', ascending = False)

#Average height by position for drafted players
ncaab[['Height', 'Position']][ncaab['Drafted'] == 'Y'].groupby('Position', as_index=False).mean().sort_index(by='Height', ascending = False)

#Average stats by position
ncaab.groupby('Position', as_index=False).mean().sort_index(by='Height', ascending = False)

#Max stats by position for drafted players
ncaab[ncaab.Drafted == 'Y'].groupby('Position').max()

ncaab[ncaab.Drafted=='N']['PPG'].max()

ncaab[ncaab.Drafted=='Y']['PPG'].max()

ncaab[['Height', 'APG'][ncaab.Drafted == 'N'].groupby('Position').max()

ncaab[[measureables]][ncaab.Drafted == 'N'].groupby('Position').max()



ncaab[['Height', 'Weight', 'Position', 'BPG', 'RPG']][ncaab['Drafted'] == 'Y'].groupby('Position', as_index=False).mean().sort_index(by='Height', ascending = False)


ncaab.groupby('Position', as_index=False).Height.max().sort_index(by='Height', ascending = False)


ncaab.groupby('Position', 'Drafted').Height.mean().sort_index(by='Height')

ncaab[[ncaab['Drafted'] == 'Y']].groupby('Position', as_index=False).mean()



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




'''
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=4.5, aspect=0.7)

# include a "regression line"
sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=4.5, aspect=0.7, kind='reg')
'''
#doesnt work
columns= ncaab.columns
sns.pairplot(ncaab[[ncaab.columns]][ncaab['Drafted'] == 'Y'], kind='reg')
sns.heatmap(ncaab.corr())


#heatmap for Drafted Y/N - next three lines
df=pd.DataFrame(ncaab)
g = sns.FacetGrid(df, col='Drafted')
g.map_dataframe(lambda data, color: sns.heatmap(data.corr(), linewidths=0))

#Parallel coordinates
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates
features = ['PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TPG', 'FG%', '3P%', 'Position']
ncaab_df = pd.DataFrame(ncaab[ncaab.Drafted=='N'][ncaab.Position!='F'][ncaab.Position!='G'], columns = features)
parallel_coordinates(data=ncaab_df, class_column='Position')

plt.figure()


features = ['PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TPG', 'FG%', '3P%', 'Position']
ncaab_df = pd.DataFrame(ncaab[ncaab.Drafted=='Y'], columns = features)
parallel_coordinates(data=ncaab_df, class_column='Position')

plt.figure()

#Parallel coordinates G Drafted/Not
features = ['PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TPG', 'FG%', '3P%', 'Drafted']
ncaab_df = pd.DataFrame(ncaab[ncaab.Position=='G'], columns = features)
parallel_coordinates(data=ncaab_df, class_column='Drafted')

###standardize axes
df.stdcolumn = (df.column - mean(df.column) ) / np.std(df.column) 

plt.figure()
