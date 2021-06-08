import datetime
import streamlit as st
import pandas as pd
import pickle

st.title("Flight price prediction module for the awesome touring company")
 
dat=st.date_input("When do you plan to take the flight?",datetime.datetime(2021,7,1))
tim = st.time_input(f"At what time are you planning to fly on {dat}?",datetime.time(7,1))
route = st.selectbox('Route options available',('Bangalore-New Delhi', 'Chennai-Kolkata', 'New Delhi-Cochin','Kolkata-Bangalore','Mumbai-Hyderabad'))
airline = st.selectbox('Preferred choice of airline',('Air Asia', 'GoAir', 'IndiGo', 'SpiceJet','Vistara'))

st.write(f'Looking for flights for {route} via {airline} on {dat} at {tim}... ')
h=pd.read_csv(r"./Flight_price/deployment/hour_calculation.csv")


def processor(dat, tim, route, airline):
    dic={'Bangalore-New Delhi':1, 'Chennai-Kolkata':2, 'New Delhi-Cochin':3,
    'Kolkata-Bangalore':4, 'Mumbai-Hyderabad':5}
    frame=pd.Series(dtype='int')
    frame['Day of month']=int(dat.day)
    frame['Month of year']=int(dat.month)
    frame['Day of week'] = int(dat.strftime("%w"))
    frame['Day of year'] = int(dat.timetuple().tm_yday)
    frame['Dep_hour'] = int(tim.hour)
    frame['Dep_minutes'] = int(tim.minute)
    frame['Dep_early morning'] = int(0<frame['Dep_hour']<=6)
    frame['Dep_morning'] = int(6<frame['Dep_hour']<=12)
    frame['Dep_noon'] = int(12<frame['Dep_hour']<=16)
    frame['Dep_evening'] = int(16<frame['Dep_hour']<=20)
    frame['Dep_night'] = int(23<frame['Dep_hour'])
    frame['IndiGo'] = int(airline=='IndiGo')
    frame['Air India'] = int(airline=='Air India')
    frame['Air Asia'] = int(airline=='Air Asia')
    frame['GoAir'] = int(airline=='GoAir')
    frame['SpiceJet'] = int(airline=='SpiceJet')
    frame['Vistara'] = int(airline=='Vistara')
    frame['Duration_hours']=list(h[(h.Airline==airline) & (h.Source==route.split('-')[0]) & (h.Destination==route.split('-')[1])]['Duration_hours'])[0]
    frame['Duration_mins']=list(h[(h.Airline==airline) & (h.Source==route.split('-')[0]) & (h.Destination==route.split('-')[1])]['Duration_mins'])[0]
    frame['route']=int(dic[route])
    frame=pd.DataFrame(frame).transpose()
    frame=frame.reindex(['Dep_hour', 'Dep_minutes', 'Duration_hours', 'Duration_mins',
       'Air Asia', 'GoAir', 'IndiGo', 'SpiceJet', 'Vistara', 'Day of month',
       'Day of week', 'Month of year', 'Day of year', 'Dep_early morning',
       'Dep_evening', 'Dep_morning', 'Dep_night', 'Dep_noon', 'route'],axis=1)
    return frame

try:

    datafr=processor(dat, tim, route, airline)

    with open('./Flight_price/deployment/model','rb') as f:
        model=pickle.load(f)

    prediction=model.predict(datafr)

    st.text(f"The estimated price for the flight is Rs. {prediction[0]}")


except:
    st.text('This particular combination of options is not available for the selected date')
