import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion, Pipeline
from datetime import datetime as dt
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
import pickle


class date_splitter(BaseEstimator,TransformerMixin):
    
    def __init__(self,Date_of_Journey):
        self.Date_of_Journey=Date_of_Journey

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):

        X['Date_of_Journey'] = X['Date_of_Journey'].apply(lambda x: dt.strptime(str(x),'%m/%d/%y'))
        X['Day of month'] = X['Date_of_Journey'].apply(lambda x: x.strftime("%d")).astype(int)
        X['Day of week'] = X['Date_of_Journey'].apply(lambda x: x.strftime("%w")).astype(int)
        X['Month of year'] = X['Date_of_Journey'].apply(lambda x: x.strftime("%m")).astype(int)
        X['Day of year'] = X['Date_of_Journey'].apply(lambda x: x.timetuple().tm_yday)

        X.drop(['Date_of_Journey'],axis=1,inplace=True)
        return X[['Day of month','Day of week','Month of year','Day of year']].values

class route(BaseEstimator,TransformerMixin):
    def __init__(self,Source,Destination):
        self.Source=Source
        self.Destination=Destination

    def fit(self,X,y=None):
        return self

    def transform(self, X, y=None):

        mapper={
        'BangaloreNew Delhi':1, 
        'ChennaiKolkata':2, 
        'New DelhiCochin':3,
        'KolkataBangalore':4, 
        'MumbaiHyderabad':5}        

        X['route'] = X['Source']+X['Destination']

        X['route'] = X['route'].map(mapper)
        X['route'] = X['route'].apply(lambda x:int(x))

        X.drop(['Source','Destination','Route','Additional_Info','Arrival_Time'],axis=1,inplace=True)
        
        return X[['route']].values

class time_trier(BaseEstimator,TransformerMixin):
    def __init__(self, Duration):
        self.Duration= Duration

    def fit(self,X,y=None):
        return self

    def transform(self, X, y=None):
             
        dur_hour = lambda x:x[:x.index("h")] if 'h' in x else 0
        dur_min = lambda x: x[x.index("m")-2:x.index("m")] if 'm' in x else 0

        X['Duration_hours'] = X.Duration.apply(dur_hour)
        X['Duration_mins'] = X.Duration.apply(dur_min)
        
        X.Duration_mins.replace({'':'0'},inplace=True)
          
        X['Duration_hours'] = X['Duration_hours'].apply(lambda x:int(x))
        X['Duration_mins'] = X['Duration_mins'].apply(lambda x:int(x))

        X.drop(['Duration'],axis=1,inplace=True)
        
        return X[['Duration_hours','Duration_mins']].values

class tod_departure(BaseEstimator,TransformerMixin):
    
    def __init__(self, Dep_Time):
        self.Dep_Time = Dep_Time

    def fit(self,X,y=None):
        return self

    def transform(self, X, y=None):
        
        hour = lambda x: x[:x.index(":")]        
        minutes = lambda x: x[x.index(":")+1:]

        X['Dep_hour'] = X.Dep_Time.apply(hour)
        X['Dep_minutes'] = X.Dep_Time.apply(minutes)

        X['Dep_minutes'] = X['Dep_minutes'].apply(lambda x: int(x))
        X['Dep_hour'] = X['Dep_hour'].apply(lambda x:int(x))
     
        tod = lambda x: 'early morning' if 0<x<=6 else('morning' if 6<x<=12 else ('noon' if 12<x<=16 else ('evening' if 16<x<=20 else 'night')))
        X['TOD'] = X.Dep_hour.map(tod)
        X.drop(['Dep_Time'],axis=1,inplace=True)
        
        return X[['TOD','Dep_minutes','Dep_hour','Airline']].values

class filters(BaseEstimator,TransformerMixin):
    def __init__(self, Total_Stops):
        self.Total_Stops=Total_Stops

    def fit(self,X,y=None):
        return self

    def transform(self, X, y=None):

        non_stop={'non-stop':1, np.nan:1, '2 stops':0, '1 stop':0, '3 stops':0,'4 stops':0}
        X.Total_Stops = X.Total_Stops.map(non_stop)
        
        X = X[X.Total_Stops==1]

        return X.values

encoder=ColumnTransformer([('airline_TOD',OneHotEncoder(
    ),
    [10,7])], remainder='passthrough')

features=FeatureUnion(

    transformer_list=[
        ('date_spliiter',date_splitter('Date_of_Journey')),
        ('route_identifier', route('Source','Destination')),
        ('timer', time_trier('Duration')),
        ('time of departure',tod_departure('Dep_Time')) ])

#Filtering only direct flights
pipe=Pipeline([('filter_hopping_flights', filters('Total_Stops'))])
#df=pd.read_csv(r'../Data/flight_price.csv')
df=pd.read_csv(r'./Flight_price/data/flight_price.csv')
dataset=pd.DataFrame(pipe.fit_transform(df))
y=dataset.iloc[:,-1]
X=dataset.iloc[:,:-1]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)
trainset = pd.concat([X_train,y_train],axis=1)
testset = pd.concat([X_test,y_test],axis=1)

dataset.columns = df.columns
trainset.columns = df.columns
testset.columns = df.columns

y = pd.DataFrame(trainset.Price)
X = trainset.drop(['Price'], axis=1)
y_test = pd.DataFrame(testset.Price)
X_test = testset.drop(['Price'], axis=1)

features.fit(X)
X = pd.DataFrame(features.transform(X))
X_test = pd.DataFrame(features.transform(X_test))
encoder.fit(X)
X = pd.DataFrame(encoder.transform(X))
X_test = pd.DataFrame(encoder.transform(X_test))

trainset = pd.concat([X,pd.DataFrame(y_train.values)],axis=1)    
testset  =pd.concat([X_test, pd.DataFrame(y_test.values)],axis=1)

#Outlier removal
trainset['std_price'] = (trainset.iloc[:,-1]-trainset.iloc[:,-1].mean())/trainset.iloc[:,-1].std()
trainset = trainset[(trainset['std_price']<3) & (trainset['std_price']>-3)]
trainset.drop(['std_price'],axis=1,inplace=True)
'''
dataset.to_csv(r'../Data/dataset.csv',index=False)
trainset.to_csv(r'../Data/trainset.csv',index=False)
testset.to_csv(r'../Data/testset.csv',index=False)
'''
dataset.to_csv(r'./Flight_price/Data/dataset.csv',index=False)
trainset.to_csv(r'./Flight_price/Data/trainset.csv',index=False)
testset.to_csv(r'./Flight_price/Data/testset.csv',index=False)

with open(r'./Flight_price/bin/features','wb') as f1:
#with open(r'../bin/features','wb') as f1:
    pickle.dump(features,f1)

with open(r'./Flight_price/bin/encoder','wb') as f2:
#with open(r'../bin/encoder','wb') as f2:
    pickle.dump(encoder,f2)
