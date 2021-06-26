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
        
        try:
            X['Date_of_Journey'] = X['Date_of_Journey'].apply(lambda x: dt.strptime(str(x), '%d-%m-%y'))
        except:
            X['Date_of_Journey'] = X['Date_of_Journey'].apply(lambda x: dt.strptime(str(x), '%Y-%m-%d %H:%M:%S'))

        #24-03-19
        
        X['Day of month'] = X.Date_of_Journey.apply(lambda x: x.strftime("%d")).astype(int)
        X['Day of week'] = X.Date_of_Journey.apply(lambda x: x.strftime("%w")).astype(int)
        X['Month of year'] = X.Date_of_Journey.apply(lambda x: x.strftime("%m")).astype(int)
        X['Day of year'] = X.Date_of_Journey.apply(lambda x: x.timetuple().tm_yday)
        
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

        df['Duration_hours'] = df.Duration.apply(dur_hour)
        df['Duration_mins'] = df.Duration.apply(dur_min)
        
        df.Duration_mins.replace({'':'0'},inplace=True)
          
        df.Duration_hours = df.Duration_hours.astype(int)
        df.Duration_mins = df.Duration_mins.astype(int)

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

        df['Dep_hour'] = df.Dep_Time.apply(hour)
        df['Dep_minutes'] = df.Dep_Time.apply(minutes)

        df.Dep_minutes = df.Dep_minutes.astype(int)
        df.Dep_hour = df.Dep_hour.astype(int)
     
        tod = lambda x: 'early morning' if 0<x<=6 else('morning' if 6<x<=12 else ('noon' if 12<x<=16 else ('evening' if 16<x<=20 else 'night')))
        df['TOD'] = df.Dep_hour.map(tod)
        X.drop(['Dep_Time'],axis=1,inplace=True)
        
        return X[['TOD','Dep_minutes','Dep_hour']].values

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
    drop=['Multiple carriers Premium economy','night']),
    ['Airline','TOD'])], remainder='passthrough')

features=FeatureUnion(

    transformer_list=[
        ('date_spliiter',date_splitter('Date_of_Journey')),
        ('route_identifier', route('Source','Destination')),
        ('timer', time_trier('Duration')),
        ('time of departure',tod_departure('Dep_Time')) ])
with open(r'./Flight_price/bin/features','wb') as f1:
    pickle.dump(features,f1)

with open(r'./Flight_price/bin/encoder','wb') as f2:
    pickle.dump(encoder,f2)

pipe=Pipeline([('filter_hopping_flights', filters('Total_Stops'))])
dataset=pd.DataFrame(pipe.fit_transform(pd.read_csv(r'./Flight_price/Data/flight_price.csv')))
y=dataset.iloc[:,-1]
X=dataset.iloc[:,:-1]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

trainset = pd.concat([X_train,y_train],axis=1)
testset = pd.concat([X_test,y_test],axis=1)

dataset.to_csv(r'./Flight_price/Data/dataset.csv')
trainset.to_csv(r'./Flight_price/Data/trainset.csv')
testset.to_csv(r'./Flight_price/Data/testset.csv')