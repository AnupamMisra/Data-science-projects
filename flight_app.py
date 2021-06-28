import datetime
import streamlit as st
import pandas as pd
import pickle
import Flight_price.src.Preprocessing
import Flight_price.src.Modelling

h=pd.read_csv(r"./Flight_price/data/hour_calculation.csv")

st.title("Flight price prediction module for the awesome touring company")
 
dat=st.date_input("When do you plan to take the flight?",datetime.date(2020, 5, 17))
dat=str(dat.strftime('%m/%d/%y'))
tim = st.time_input(f"At what time are you planning to fly on {dat.strftime('%d/%m%y')}?",datetime.time(7,1))
tim = str(tim.strftime('%H:%M'))


airline = st.selectbox('Preferred choice of airline',('IndiGo', 'Air India', 'SpiceJet',
       'GoAir', 'Vistara', 'Air Asia'))

sourcelist=h[h['Airline']==airline].Source.values.tolist()

src = st.selectbox('Source',sourcelist)

destinationlist=h[(h['Airline']==airline) & (h['Source']==src)].Destination.values.tolist()

dest= st.selectbox('Destination',destinationlist)

st.write(f'Looking for flights from {src} to {dest} via {airline} on {dat} at {tim}... ')

def duration(src,dest,airline):

    identified = h[(h['Source']==src) & (h['Destination']==dest) & (h['Airline']==airline)]
    dur=identified.Duration.values[0]
    return dur

def processor(dat, tim, airline,src,dest):

    airline=airline
    doj=dat
    src=src
    dest=dest
    route='asdf'
    dep_time=tim
    arr_time='10:20'

    
    dur=duration(src,dest,airline)
    addi='non-stop'
    stops=1

    data=[airline,doj,src,dest,route,dep_time,arr_time,dur,addi,stops]
    vase=pd.DataFrame(pd.Series(data,index=['Airline',
 'Date_of_Journey',
 'Source',
 'Destination',
 'Route',
 'Dep_Time',
 'Arrival_Time',
 'Duration',
 'Total_Stops',
 'Additional_Info'])).transpose()
    return vase

try:

    with open('../bin/feats','rb') as f1:
        feats=pickle.load(f1)
        
    with open('../bin/code','rb') as f2:
        code=pickle.load(f2)

    with open('../bin/model','rb') as f3:
        model=pickle.load(f3)

    input=processor(dat, tim, airline,src,dest)
    input=feats.transform(input)
    input=code.transform(input)
    y=model.predict(input)[0]

    time=duration(src,dest,airline)

    st.text(f"The estimated price for the flight is Rs. {y}")
    st.text(f"Flight duration: {time}")
    st.text("The prices won't be accurate as the model was trained on pre-Covid data.")

except:

    st.write('Sorry the flights cannot be loaded for the selected combination')