from datetime import datetime
import streamlit as st
st.write("Hello world. This is my first streamlit project")
dat=st.date_input("When do you want to take flight",datetime.date(2021,5,19))
st.write(f'Date entered: {dat}')