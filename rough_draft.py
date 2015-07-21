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

ncaab.plot(x='Height', y='Weight', kind='scatter', alpha=0.3)