# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 21:06:35 2015

@author: Danny
"""

# imports
import pandas as pd
import numpy as np

# read in the CSV file from a URL
ncaab = pd.read_csv('https://raw.githubusercontent.com/dansnacks/SF_DAT_15_WORK/master/project_data.csv')

ncaab.dropna()

ncaab[['Player', 'BPG', 'Drafted']][ncaab.Position=='C'].sort_index(by='Drafted')


def color_drafted(yesno):
    if yesno == 'Y':
        return 'b'
    else:
        return 'r'

colors = ncaab.Drafted.apply(color_drafted)
colors
#see if pattern for height/weight and draft status
ncaab.plot(x='Height', y='Weight', kind='scatter', alpha=0.3, c=colors)

def color_position(position):
    if position == 'C':
        return 'r'
    elif position == 'PG' or position == 'SG' or position == 'G':
        return 'b'
    elif position == 'PF' or position == 'SF' or position == 'F':        
        return 'g'    
    else:
        return 'y'

colorposition = ncaab.Position.apply(color_position)
ncaab.plot(x='Height', y='Weight', kind='scatter', alpha=0.3, c=colorposition)


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


sns.pairplot(ncaab)
'''

'''
from pandas.tools.plotting import parallel_coordinates
# I'm going to convert to a pandas dataframe
# Using a snippet of code we learned from one of Kevin's lectures!
features = [name[:-5].title().replace(' ', '') for name in iris.feature_names]
iris_df = pd.DataFrame(iris.data, columns = features)
iris_df['Name'] = iris.target_names[iris.target]
parallel_coordinates(data=iris_df, class_column='Name', 
                     colors=('#FF0054', '#FBD039', '#23C2BC'))
'''