import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer as CT
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split 
import pickle
 
df = pd.read_csv(r"./Customer_churn/Data/Telco-Customer-Churn.csv")
df.drop(["customerID"], axis=1, inplace=True)

df.SeniorCitizen=df.SeniorCitizen.apply(lambda x: str(x))

binary_feat = df.nunique()[df.nunique() == 2].keys().tolist()
numeric_feat = [col for col in df.select_dtypes([np.float64,np.int64]).columns.tolist() if col not in binary_feat]
categorical_feat = [ col for col in df.select_dtypes('object').columns.to_list() if col not in binary_feat + numeric_feat ]

binary_feat.remove('Churn')
target=df.Churn

nos=["No phone service", "No"]
nos.extend( ["No internet service"]*6)
nos.extend(['Two year','Mailed check'])

preprocessing = CT(
                    transformers=[
                        ('numeric_scaling', MinMaxScaler(), numeric_feat),
                        ('categorical_dummies', OneHotEncoder(drop=nos), categorical_feat),
                        ('binary_binarizing', OneHotEncoder(drop='if_binary'), binary_feat)
                                 ],
                        remainder='drop',
                    n_jobs=-1
                    )

X_train, X_test, y_train, y_test = train_test_split(df.drop(['Churn'],axis=1),target, test_size=0.2, random_state=123456)

X_train = pd.DataFrame(preprocessing.fit_transform(X_train))
X_test = pd.DataFrame(preprocessing.transform(X_test))

train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

with open(r'./Customer_churn/bin/preprocessing','wb') as r:
    pickle.dump(preprocessing,r) 

train.to_csv(r'./Customer_churn/data/train.csv')
test.to_csv(r'./Customer_churn/data/test.csv')
