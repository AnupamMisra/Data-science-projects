
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, recall_score as R
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_selection import chi2
import pickle

#Notebook options

pd.options.display.max_columns =100
warnings.filterwarnings('ignore')

train = pd.read_csv(r"./Customer_churn/Data/train.csv")
test = pd.read_csv(r"./Customer_churn/Data/test.csv")

train.drop(['Unnamed: 0'],axis=1,inplace=True)
test.drop(['Unnamed: 0'],axis=1,inplace=True)

l=LabelBinarizer()
y = l.fit_transform(train.Churn)
X = train.iloc[:,:-1]

with open (r'./Customer_churn/binaries/preprocessing','rb') as t:
    scaler=pickle.load(t)

X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.2, stratify=y, random_state=12345)

X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_valid = pd.DataFrame(scaler.transform(X_valid))

ch=pd.DataFrame(chi2(X_train, y_train)).transpose()
ch.columns=['Chi squared value','p-value']
ch['p-value'] = ch['p-value'].apply(lambda x: float(x))
to_drop = ch[ch['p-value']>0.05].index.tolist()

X_train.drop(to_drop,axis=1,inplace=True)
X_valid.drop(to_drop,axis=1,inplace=True)

