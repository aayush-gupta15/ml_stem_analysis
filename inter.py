# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#STAGE 1

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

#STAGE 3 Using Backword Elimination method and choosing p<0.05 only i choose 2,3,4,5,6,7,8 column

X=imp.iloc[:,[2,4,5,6,7,8]].values
y=imp.iloc[:,10].values



from sklearn.preprocessing import Imputer
imputer1 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer1 = imputer1.fit(X[:, 1:4])
X[:, 1:4] = imputer1.transform(X[:, 1:4])


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 0] = labelencoder_X1.fit_transform(X[:, 0])
labelencoder_X2 = LabelEncoder()
X[:, 5] = labelencoder_X2.fit_transform(X[:, 5])
onehotencoder = OneHotEncoder(categorical_features = [0,5])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)

#roc=0.796428




