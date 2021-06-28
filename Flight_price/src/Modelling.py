import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope
from functools import partial

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

def optimize(params, X,y):
    
    model=RandomForestRegressor(**params)
    k=KFold()
    mapes=[]
    
    for idx in k.split(X=X,y=y):
        train_idx,test_idx=idx[0],idx[1]
        
        xtrain=X[train_idx]
        ytrain=y[train_idx]

        xtest=X[test_idx]
        ytest=y[test_idx]

        model.fit(xtrain,ytrain)
        preds=model.predict(xtest)
        
        fold_acc = mean_absolute_percentage_error(ytest, preds)
        mapes.append(fold_acc)
    
    return np.mean(mapes)
    
if __name__=='__main__':
    trainset = pd.read_csv(r'./Flight_price/Data/trainset.csv')
    testset = pd.read_csv(r'./Flight_price/Data/testset.csv')
    X_train, y_train = trainset.iloc[:,:-1], trainset.iloc[:,-1]
    X_test, y_test = testset.iloc[:,:-1], testset.iloc[:,-1]

    rf_space = {
            "max_depth": scope.int(hp.quniform('max_depth',1,50,1)),
            "n_estimators": scope.int(hp.quniform('n_estimators',10,500,1)),
            "max_features": hp.uniform('max_features',0.01,1)
                }

    

    optimization_function = partial(optimize, X=X_train.values, y=y_train.values)

    trials = Trials()

    result = fmin(
                fn=optimization_function,
                space=rf_space,
                max_evals=15,
                trials=trials,
                algo=tpe.suggest
            ) 

print(result)
model = RandomForestRegressor(max_depth=int(result['max_depth']), max_features=result['max_features'], n_estimators=int(result['n_estimators']))

dataset = pd.read_csv(r'./Flight_price/Data/dataset.csv')

with open('./Flight_price/bin/features.pkl','rb') as f1:
    features=pickle.load(f1)

with open('./Flight_price/bin/encoder.pkl','rb') as f2:
    encoder=pickle.load(f2)
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

feats=features.fit(X)
X=feats.transform(X)
code=encoder.fit(X)
X=code.transform(X)
model.fit(X,y)

with open(r'./Flight_price/bin/feats','wb') as f1:
    pickle.dump(feats,f1)

with open(r'./Flight_price/bin/code','wb') as f2:
    pickle.dump(code,f2)

with open(r'./Flight_price/bin/model','wb') as f3:
    pickle.dump(model,f3)

