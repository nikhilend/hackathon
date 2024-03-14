import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
# age =25
# experience = 2
# income=30
# zip=641010
# Family=4
# avgcc=3
# edu=2
# mortgage=0
# sec=0
# cd=0
# net=0
# cc=0
# personalloan='false'
# creditcard='false'
#
# personalinput = [age, experience, income, zip, Family, avgcc, edu, mortgage, sec, cd, net, cc]
#
# ccinput = [age, experience, income, zip, Family, avgcc, edu, mortgage, sec, cd, net]
# input_datapl = pd.DataFrame([personalinput], columns=['Age', 'Experience', 'Income in Thousaunds', 'ZIP Code', 'Family',
#                                                       'CCAvg in thousands', 'Education', 'Mortgage',
#                                                       'SecuritiesAccount', 'CDAccount', 'Online(opted Netbanking)',
#                                                       'CreditCard'])
# input_datacc = pd.DataFrame([ccinput],
#                             columns=['Age', 'Experience', 'Income in Thousaunds', 'ZIP Code', 'Family',
#                                      'CCAvg in thousands', 'Education', 'Mortgage', 'SecuritiesAccount',
#                                      'CDAccount', 'Online(opted Netbanking)'])
# modelp = joblib.load('personalloan.pkl')
# modelc = joblib.load('creditcard.pkl')
# predictionp = modelp.predict(input_datapl)[0]
# predictionc = modelc.predict(input_datacc)[0]
# # output = predictionp +'-'+predictionp
# print(predictionc)
#
#
# if personalloan == 'false' and predictionp==1: print('personal loan can be suggested')
# if creditcard == 'false' and predictionc==1: print('credit card can be suggested')
# print(predictionc)



# test 2
# new_data = pd.read_csv('input.csv')
# model = joblib.load('personalloan.pkl')
# predictions = model.predict(new_data)
# predictions_df = pd.DataFrame(predictions, columns=['Predicted'])
# result_df = pd.concat([new_data, predictions_df], axis=1)
# result_df.to_csv('input.csv', index=False)

#test 3
import os
import openai
import streamlit as st

# Function to generate a response from the chatbot
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


# Streamlit UI
def main():
    st.title("Chatbot")

    # Text input for user input
    user_input = st.text_input("Enter your message:")

    # Button to submit user input
    if st.button("Send"):
        # Display user input
        st.write("You:", user_input)

        # Generate and display chatbot response
        bot_response = generate_response(user_input)
        st.write("Bot:", bot_response)


if __name__ == "__main__":
    main()