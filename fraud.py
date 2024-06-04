import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
from PIL import Image

# page title
st.title(":orange[Credit Card Fraud Detection with ANN]")

df0 = pd.read_csv('creditcard.csv')
df = df0.copy()
df.drop("Class", axis=1, inplace=True)

st.subheader("⚡ Select the type of visual from the dropdown below")

queries = ["Proportion of Non-Fraud and Fraud Transactions",
           "Distribution of Amount",
           "Distribution of Time",
          ]
selection = st.selectbox("   ", options=queries)

if selection=="Proportion of Non-Fraud and Fraud Transactions":
      fig1=px.pie(data_frame=df0,values=df0.Class.value_counts(normalize=True),
        names = ["Non Fraud","Fraud"], color_discrete_sequence=['green','red'],
             title='Proportion of Non-Fraud and Fraud Transactions')
      st.plotly_chart(fig1, use_container_width=True)
   

elif selection=="Distribution of Amount": 
     amount_counts = df0['Amount'].value_counts().reset_index()
     amount_counts.columns = ['Amount', 'Count']
     amount_counts['Class'] = df0['Class']

     fig2 = px.histogram(amount_counts, x='Amount', y='Count',color='Class',nbins=20,
             title='Distribution of Amount Values', color_discrete_sequence=['green','red'])
     fig2.update_xaxes(range=[0,26000])
     st.plotly_chart(fig2, use_container_width=True)

else:
      amount_counts = df0['Time'].value_counts().reset_index()
      amount_counts.columns = ['Time', 'Count']
      amount_counts['Class'] = df0['Class']
     

      fig3 = px.histogram(amount_counts, x='Time', y='Count',nbins=40,color='Class',
             title='Distribution of Time Values', color_discrete_sequence=['green','red'])
      fig3.update_xaxes(range=[0,173000])
      st.plotly_chart(fig3, use_container_width=True)
  


st.subheader("⚡ Use the sidebar menu to change the prediction parameters")

# Main Image
image = Image.open("image/security.jpg")
#st.image(image, use_column_width=True)

# Sidebar title
st.sidebar.title('Credit Card Fraud Detection')

# Sidebar image
st.sidebar.image(image, use_column_width=True)

# Side bar user inputs
V14=st.sidebar.slider("1st parameter (V14)", -19.0,11.0, step=0.5 )
V10=st.sidebar.slider("2nd parameter (V10)", -25.0,24.0, step=0.5)
V17=st.sidebar.slider("3rd parameter (V17)", -26.0,10.0, step=0.5)
V4=st.sidebar.slider("4th parameter (V4)", -6.0, 17.0, step=0.5)
V12=st.sidebar.slider("5th parameter (V12)", -19.0,8.0, step=0.5)


# Converting user inputs to dataframe 
my_dict = {'V14': V14,
           'V10': V10,
           'V17': V17,
           'V4': V4,
           'V12': V12
}
df2 = pd.DataFrame.from_dict([my_dict])
df2.index = [''] * df2.shape[0]


st.success("◽ Current prediction parameters are:")
st.table(df2)

# Loading the model(s) to make predictions
loaded_model=pickle.load(open("xgb_model_fraud_detection.pkl","rb"))


# defining the function which will make the prediction using the data
def get_prediction(model, input_data):
	prediction = model.predict(input_data)
	return prediction


st.subheader("⚡ Press the 'Predict' button below to get a prediction")

if st.button("Predict"):
    result = get_prediction(loaded_model, df2)[0]
    if result == 0:
        result = "legit"
        proba=loaded_model.predict_proba(df2)[:,0]
        probability= round(proba[0]*100, 2)
        st.success(f"◽ My prediction is :   **The transaction is likely with % {probability} to be {result}**")
        st.image(Image.open("image/check.png"))

    elif result == 1:
        result = "fraud"
        proba=loaded_model.predict_proba(df2)[:,1]
        probability= round(proba[0]*100, 2)
        st.success(f"◽ My prediction is :   **The transaction is likelywith % {probability} to be {result}**")
        st.image(Image.open("image/fraud_alert.png"))