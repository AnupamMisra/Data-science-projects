from datetime import datetime
import streamlit as st
st.write("Hello world. This is my first streamlit project")
dat=st.date_input("When are you planning to take the flight",datetime(2021,5,19))
st.write(f'Date entered: {dat}')

src=st.text_input("Enter source")
dest=st.text_input("Enter destination")
airline = st.selectbox('Preferred airline',('Air Asia', 'GoAir', 'IndiGo', 'Multiple carriers','SpiceJet','Vistara'))
st.write(f'Looking for flights from {src} to {dest} via {airline} on {dat}... ')
