import streamlit as st
import pandas as pd
import base64
from io import BytesIO
import pickle


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
    vals = [0,'Male/Female','Yes/No','Yes/No','Yes/No',123,'Yes/No','Yes/No/No phone service','DSL/Fibre optic/No','Yes/No/No internet sevice','Yes/No/No internet sevice','Yes/No/No internet sevice','Yes/No/No internet sevice','Yes/No/No internet sevice','Yes/No/No internet sevice','Month-to-month/One year/Two year','Yes/No','Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)',123,123]
    d= pd.Series(vals, index=['CustomerID','gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure','PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity','OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV','StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod','MonthlyCharges', 'TotalCharges'])
    d=pd.DataFrame(d).transpose()
    
    st.markdown(get_table_download_link(d), unsafe_allow_html=True)
    uploaded_file=st.file_uploader("Please upload data in the provided format")
    
    if uploaded_file is not None:
        dff=pd.read_excel(uploaded_file)
        dff=dff[dff.CustomerID!=0]
        return dff

    else:
        st.text("Waiting for your input...")    

def preprocess(df):

    X=pd.DataFrame()
    X['gender'] = (df.gender=='Male').astype('int')
    X['SeniorCitizen'] = (df.SeniorCitizen=='Yes').astype('int')
    X['Partner'] =( df.Partner=='Yes').astype('int')
    X['Dependents'] = (df.Dependents=='Yes').astype('int')
    X['PhoneService'] = (df.PhoneService=='Yes').astype('int')
    X['PaperlessBilling'] = (df.PaperlessBilling=='Yes').astype('int')
    X['MultipleLines_No'] = (df.MultipleLines=="No").astype('int')
    X['MultipleLines_Yes'] = (df.MultipleLines=="Yes").astype('int')
    X['MultipleLines_No phone service'] = (df.MultipleLines=="No phone service").astype('int')
    X['InternetService_Fiber optic'] = (df.InternetService=="Fibre optic").astype('int')
    X['InternetService_No'] =( df.InternetService=="No").astype('int')
    X['InternetService_DSL'] = (df.InternetService=="DSL").astype('int')
    X['OnlineSecurity_No'] = (df.OnlineSecurity=="No").astype('int')
    X['OnlineSecurity_No internet service'] = (df.OnlineSecurity=="No internet service").astype('int')
    X['OnlineSecurity_Yes'] =( df.OnlineSecurity=="Yes").astype('int')
    X['OnlineBackup_No'] = (df.OnlineBackup=='No').astype('int')
    X['OnlineBackup_No internet service'] = (df.OnlineBackup=='No internet service').astype('int')
    X['OnlineBackup_Yes'] = (df.OnlineBackup=='Yes').astype('int')
    X['DeviceProtection_No'] = (df.DeviceProtection=="No").astype('int')
    X['DeviceProtection_No internet service'] = (df.DeviceProtection=="No internet service").astype('int')
    X['DeviceProtection_Yes'] =( df.DeviceProtection=="Yes").astype('int')
    X['TechSupport_No internet service'] = (df.TechSupport=="No internet service").astype('int')
    X['TechSupport_Yes']= (df.TechSupport=="Yes").astype('int')
    X['TechSupport_No']= (df.TechSupport=="No").astype('int')
    X['StreamingTV_No'] =( df.StreamingTV=="No").astype('int')
    X['StreamingTV_No internet service'] = (df.StreamingTV=="No internet service").astype('int')
    X['StreamingTV_Yes'] = (df.StreamingTV=="Yes").astype('int')
    X['StreamingMovies_No'] = (df.StreamingMovies=="No").astype('int')
    X['StreamingMovies_No internet service'] = (df.StreamingMovies=="No internet service").astype('int')
    X['StreamingMovies_Yes'] = (df.StreamingMovies=="Yes").astype('int')
    X['Contract_Month-to-month']=(df.Contract=='Month-to-month').astype('int')
    X['Contract_One year']=(df.Contract=='One year').astype('int')
    X['Contract_Two year']=(df.Contract=='Two year').astype('int')
    X['PaymentMethod_Bank transfer (automatic)'] = (df.PaymentMethod=="Bank transfer (automatic)").astype('int')
    X['PaymentMethod_Credit card (automatic)'] = (df.PaymentMethod=="Credit card (automatic)").astype('int')
    X['PaymentMethod_Electronic check'] = (df.PaymentMethod=="Electronic check").astype('int')
    X['PaymentMethod_Mailed check'] = (df.PaymentMethod=="Mailed check").astype('int')

    with open('../deployment/scalar','rb') as a:
        scalar=pickle.load(a)

    X[['tenure','MonthlyCharges','TotalCharges']] = scalar.transform(df[['tenure','MonthlyCharges','TotalCharges']])
    df.drop(['MultipleLines', 'InternetService','OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies', 'Contract','PaymentMethod'],axis=1,inplace=True)
    return X
 
if __name__=='__main__':
    
    st.title("Phone company's churn prediction module for the Martian Sapiens")

    dff=uploader()
    if dff is not None:
        df=dff.drop(['CustomerID'],axis=1)
        custID=dff['CustomerID']
        
        c=st.slider("Cost of churn", min_value=500, max_value=10000,step=500)
        e=st.slider("Cost of preventing churn", min_value=0, max_value=5000,step=50)

        model_frame=preprocess(df)

        with open('../deployment/rf','rb') as a:
            rf=pickle.load(a)
        with open('../deployment/lr','rb') as a:
            lr=pickle.load(a)
        with open('../deployment/g','rb') as a:
            g=pickle.load(a)            

        y=rf.predict(model_frame)
        out=pd.DataFrame(y,index=custID)
        out.columns=['Churn']
        out.index.name='Customer ID'
        st.write(out)

         # Do overall analysis   
