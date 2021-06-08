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
        df=pd.read_excel(uploaded_file)
        df=df[df.CustomerID!=0]
        return df

    else:
        st.text("Waiting for your input...")    

def preprocess(df):

    X['gender'] = df.gender=='Male'
    X['SeniorCitizen'] = df.SeniorCitizen=='Yes'
    X['Partner'] = df.Partner=='Yes'
    X['Dependents'] = df.Dependents=='Yes'
    X['PhoneService'] = df.PhoneService=='Yes'
    X['PaperlessBilling'] = df.PaperlessBilling=='Yes'
    X['MultipleLines_No'] = df.MultipleLines=="No"
    X['MultipleLines_Yes'] = df.MultipleLines=="Yes"
    X['MultipleLines_No phone service'] = df.MultipleLines=="No phone service"
    X['InternetService_Fiber optic'] = df.InternetService=="Fibre optic"
    X['InternetService_No'] = df.InternetService=="No"
    X['InternetService_DSL'] = df.InternetService=="DSL"
    X['OnlineSecurity_No'] = df.OnlineSecurity=="No"
    X['OnlineSecurity_No internet service'] = df.OnlineSecurity=="No internet service"
    X['OnlineSecurity_Yes'] = df.OnlineSecurity=="Yes"
    X['OnlineBackup_No'] = df.OnlineBackup=='No'
    X['OnlineBackup_No internet service'] = df.OnlineBackup=='No internet service'
    X['OnlineBackup_Yes'] = df.OnlineBackup=='Yes'
    X['DeviceProtection_No'] = df.DeviceProtection=="No"
    X['DeviceProtection_No internet service'] = df.DeviceProtection=="No internet service"
    X['DeviceProtection_Yes'] = df.DeviceProtection=="Yes"
    X['TechSupport_No internet service'] = df.TechSupport=="No internet service"
    X['TechSupport_Yes']= df.TechSupport=="Yes"
    X['TechSupport_No']= df.TechSupport=="No"
    X['StreamingTV_No'] = df.StreamingTV=="No"
    X['StreamingTV_No internet service'] = df.StreamingTV=="No internet service"
    X['StreamingTV_Yes'] = df.StreamingTV=="Yes"
    X['StreamingMovies_No'] = df.StreamingMovies=="No"
    X['StreamingMovies_No internet service'] = df.StreamingMovies=="No internet service"
    X['StreamingMovies_Yes'] = df.StreamingMovies=="Yes"
    X['Contract_Month-to-month']=df.Contract=='Month-to-month'
    X['Contract_One year']=df.Contract=='One year'
    X['Contract_Two year']=df.Contract=='Two year'
    X['PaymentMethod_Bank transfer (automatic)'] = df.PaymentMethod=="Bank transfer (automatic)"
    X['PaymentMethod_Credit card (automatic)'] = df.PaymentMethod=="Credit card (automatic)"
    X['PaymentMethod_Electronic check'] = df.PaymentMethod=="Electronic check"
    X['PaymentMethod_Mailed check'] = df.PaymentMethod=="Mailed check"

    with open('../deployment/scalar','rb') as a:
        scalar=pickle.load(a)

    X[['tenure','MonthlyCharges','TotalCharges']] = scalar.transform(df[['tenure','MonthlyCharges','TotalCharges']])

    return X
 
if __name__=='__main__':
    
    st.title("Phone company's churn prediction module for the Martian Sapiens")

    df=uploader()
    if df is not None:
        X=df.drop(['CustomerID'],axis=1)
        custID=df['CustomerID']

        c=st.slider("Cost of churn", min_value=500, max_value=10000,step=500)
        e=st.slider("Cost of preventing churn", min_value=0, max_value=5000,step=50)

        model_frame=preprocess(X)

        with open('../deployment/rf','rb') as a:
            rf=pickle.load(a)
        with open('../deployment/lr','rb') as a:
            lr=pickle.load(a)
        with open('../deployment/g','rb') as a:
            g=pickle.load(a)            

        y=rf.predict(model_frame)
        st.write(y)
