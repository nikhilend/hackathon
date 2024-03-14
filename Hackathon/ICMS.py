#region headers
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import openai
#endregion
#region azure implementation
def generate_response(input_text):

    openai.api_type = "azure"
    openai.api_base = "https://testingkey.openai.azure.com/"
    openai.api_version = "2023-09-15-preview"
    openai.api_key = "35f96eb3ffc04868981139be478f99fa"

    response = openai.Completion.create(
        engine="TestingChatModel",
        prompt=input_text,
        temperature=0.2,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1,
        stop=None)
    return response['choices'][0]['text']

#endregion

#region Page Design
st.title("Intelligent Client Management System")
mde = st.radio(
    "Select Mode",
    ('Form', 'Export'))
#region Form mode
if mde == "Form":
    #region inputs
    age = st.slider('Age of the Customer', 0, 130, 25)
    experience = st.slider('experience in years with bank', 0, 130, 5)
    income = st.slider('income of the Customer(in thousands)', 0, 500, 25)
    zip = st.number_input('zip code')
    Family = st.slider('Total Family members', 1, 10, 2)
    avgcc = st.slider('average Monthly expenditure(in thousands)', 0, 500, 25)
    education = st.radio(
        "Highest education level of customer",
        ('Higher secondary', 'under graduate','post graduate'))
    mortgage = st.slider('mortage(in thousands)', 0, 500, 25)
    security = st.toggle('Has security account')
    cdaccount = st.toggle('has CD account')
    onlinebanking = st.toggle('opted online banking?')
    creditcard = st.toggle('has credit card?')
    personalloan = st.toggle('has personal loan?')
    text='i will give you some details about a person, please give some banking service suggestions we can provide him age-'+str(age)+', years related with bank-'+str(experience)+', Income in Thousaunds-'+str(income)+', ZIP code-'+str(zip)+',family members-'+str(Family)+',credit card usage in average-'+str(avgcc)+',education- '+education+',mortage-'+str(mortgage)
    #endregion
    #region data preparation and passing
    if education=="Higher secondary": edu=1
    else :
        if education=="under graduate": edu=2
        else:
            edu=3
    sec = int(security == 'true')
    cd=int(cdaccount == 'true')
    net = int(onlinebanking == 'true')
    cc=int(creditcard == 'true')
    pl=int(personalloan == 'true')
    personalinput = [age, experience, income,zip,Family,avgcc,edu,mortgage,sec,cd,net,cc]

    ccinput = [age, experience, income,zip,Family,avgcc,edu,mortgage,sec,cd,net]
    input_datapl = pd.DataFrame([personalinput], columns=['Age', 'Experience', 'Income in Thousaunds','ZIP Code','Family','CCAvg in thousands','Education','Mortgage','SecuritiesAccount','CDAccount','Online(opted Netbanking)','CreditCard'])
    input_datacc= pd.DataFrame([ccinput],
                                columns=['Age', 'Experience', 'Income in Thousaunds', 'ZIP Code', 'Family',
                                         'CCAvg in thousands', 'Education', 'Mortgage', 'SecuritiesAccount',
                                         'CDAccount', 'Online(opted Netbanking)'])
    modelp = joblib.load('personalloan.pkl')
    modelc = joblib.load('creditcard.pkl')
    predictionp = modelp.predict(input_datapl)[0]
    predictionc = modelc.predict(input_datacc)[0]
    output=' '
    #endregion
    #region prediction
    if st.button('Predict'):
        if pl == 0 and predictionp==1: output+='personal loan can be suggested'
        if cc == 0 and  predictionc==1 :output+='credit card can be suggested'

        if output==' ':
            st.info("unable to predict services with certainity", icon="ℹ️")
        else:
            st.info(output, icon="ℹ️")
        bot_response = generate_response(text)
        st.write("services that could be offered:", bot_response)

    #endregion

#endregion


#region bulk mode
else:
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        if st.button('Predict'):
           # new_data = pd.read_csv(uploaded_file)
            new_data = pd.read_csv(uploaded_file)
            new_data_without_last_column = new_data.iloc[:, :-1]
            model = joblib.load('personalloan.pkl')
            model2 = joblib.load('creditcard.pkl')
            predictions = model.predict(new_data)
            predictions2 = model.predict(new_data)
           # predictions2 = model.predict(new_data_without_last_column)
            predictions_df = pd.DataFrame(predictions, columns=['Personal loan acceptance'])
            predictions_df2 = pd.DataFrame(predictions2, columns=['credit card acceptance'])
            result_df = pd.concat([new_data, predictions_df], axis=1)
            result_df = pd.concat([result_df, predictions_df2], axis=1)
            #st.info(result_df, icon="ℹ️")
            result_df.to_csv('output.csv', index=False)
            st.info("done", icon="ℹ️")


#endregion

#endregion





