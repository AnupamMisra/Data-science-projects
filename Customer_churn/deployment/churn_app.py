import streamlit as st
import pandas as pd


gender=st.selectbox('Gender',('Male','Female'))
senior=st.checkbox('Senior citizen')
partner=st.checkbox('In a relationship?')
dependents=st.checkbox('Do you have dependents?')
phone=st.checkbox('Phone service',('Yes','No'))
if phone:
    multiple=st.checkbox('Multiple lines?')
else: multiple=-1

internet=st.selectbox('Internet service',('No','DSL', 'Fiber optic'))

if internet!='No':

    security=st.checkbox('Online security')
    backup=st.checkbox('Online backup')
    protection=st.checkbox('Device protection')
    support=st.checkbox('Tech support')
    tv=st.checkbox('Streaming TV?')
    movies=st.checkbox('Streaming movies')

else:
    security=-1
    backup=-1
    protection=-1
    support=-1
    tv=-1
    movies=-1

contract=st.selectbox('Contract duration',('Month-to-month', 'One year', 'Two year'))
paperless=st.checkbox('Paperless billing?')
payment=st.selectbox('Payment mode',('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))

tenure=st.slider('Tenure(months)', min_value=0, max_value=1000,step=1)
monthly=st.slider('Monthly charges', min_value=100, max_value=1000,step=25)
total=st.slider('Total charges', min_value=100, max_value=5000,step=50)

'''
def preprocess(X):
    lb=LabelBinarizer()

    binary_feat = X.nunique()[X.nunique() == 2].keys().tolist()
    numeric_feat = [col for col in X.select_dtypes(['float','int']).columns.tolist() if col not in binary_feat]
    categorical_feat = [ col for col in X.select_dtypes('object').columns.to_list() if col not in binary_feat + numeric_feat ]

    #le = LabelEncoder()
    for i in binary_feat:
    X[i] = lb.fit_transform(X[i])

    X = pd.get_dummies(X, columns=categorical_feat)
    sc=StandardScaler()
    X = sc.fit_transform(X)

    X=pd.DataFrame(X)
    return X

'''