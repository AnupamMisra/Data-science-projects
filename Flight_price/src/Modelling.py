import pandas as pd
import numpy as np
import pickle

trainset = pd.read_csv(r'./Flight_price/Data/trainset.csv')
testset = pd.read_csv(r'./Flight_price/Data/testset.csv')

X_train, y_train = trainset.iloc[:,:-1], trainset.iloc[:,-1]
X_test, y_test = testset.iloc[:,:-1], testset.iloc[:,-1]

