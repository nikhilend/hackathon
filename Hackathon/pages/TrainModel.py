import os
import openai
import streamlit as st

st.title("Machine Learning Model trainer for data driven recommendation engine")
mde = st.radio(
    "Select service that needs a model to be trained",
    ('Personal loan', 'credit card'))
if mde == "Personal loan":
    uploaded_file = st.file_uploader("Choose a xlsx file with required data")
    st.text('please Make sure that the data is in the format:')
    st.text('Age,Experience,Income in Thousaunds,ZIP Code,Family,CCAvg in thousands,Education,Mortgage,SecuritiesAccount,CDAccount,Online(opted Netbanking),CreditCard,PersonalLoan accepted Y/N')