import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.feature_selection import chi2
import pickle

train = pd.read_csv(r"./Customer_churn/data/train.csv")
test = pd.read_csv(r"./Customer_churn/data/test.csv")

X = train.iloc[:,:-1]
X_test = test.iloc[:,:-1]

y=train.iloc[:,-1]
y_test=test.iloc[:,-1]

ch=pd.DataFrame(chi2(X, y)).transpose()
ch.columns=['Chi squared value','p-value']
ch['p-value'] = ch['p-value'].apply(lambda x: float(x))
to_drop = ch[ch['p-value']>0.05].index.tolist()
to_drop=[str(x) for x in to_drop]
X.drop(to_drop,axis=1,inplace=True)
X_test.drop(to_drop,axis=1,inplace=True)



