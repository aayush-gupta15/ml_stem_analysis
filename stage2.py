#STAGE 2

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#STAGE 1
def uniplot(a,name):
    a[name].value_counts().plot.bar()


# Importing the dataset
dataset = pd.read_excel('2010 Federal STEM Education Inventory Data Set.xls',skiprows=1)
imp=dataset.iloc[:-1,[0,1,2,3,5,6,7,8,9,10]]
funding=dataset.iloc[:-1,6:8].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(funding[:, 0:2])
funding[:, 0:2] = imputer.transform(funding[:, 0:2])

x=funding[:,0]
y=funding[:,1]

tag=[]
for i in range(0,252):
    tag.append((y[i]-x[i])/x[i])

for i in range(0,252):
     if(tag[i]<0):
        tag[i]=0
     else:
        tag[i]=1

imp['tag']=tag
#PLOT
t=list(dataset.columns.values)
for i in range(2,256):
    if((i==4)or(i==6)or(i==7)or(i==8)):
        continue
    else:    
        uniplot(dataset,t[i])




