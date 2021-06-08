import streamlit as st
import pandas as pd
import base64
from io import BytesIO

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1',index=False)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    #return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="churn_template.csv">Download template for uploading</a>' # decode b'abc' => abc
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="churn_template.csv">Download template for uploading</a>' # decode b'abc' => abc

def uploader():
    vals = [123,'Male/Female','Yes/No','Yes/No','Yes/No',123,'Yes/No','Yes/No/No phone service','DSL/Fibre optic/No','Yes/No/No internet sevice','Yes/No/No internet sevice','Yes/No/No internet sevice','Yes/No/No internet sevice','Yes/No/No internet sevice','Yes/No/No internet sevice','Month-to-month/One year/Two year','Yes/No','Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)',123,123]
    d= pd.Series(vals, index=['CustomerID','gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure','PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity','OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV','StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod','MonthlyCharges', 'TotalCharges'])
    d=pd.DataFrame(d).transpose()
    
    st.markdown(get_table_download_link(d), unsafe_allow_html=True)
    uploaded_file=st.file_uploader("Please upload data in the provided format")
    
    if uploaded_file is not None:
        df=pd.read_excel(uploaded_file, skiprows=1)
        return df

    else:
        st.text("Waiting for your input...")    

if __name__=='__main__':
    
    st.title("Customer churn prediction module")

    df=uploader()
    if df is not None:
        st.write(df.iloc[0,:])


