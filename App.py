### STEP 1:  IMPORT LIBRARIES FROM REQUIREMENTS.TXT

import numpy as np
import pandas as pd
import streamlit as st
import sklearn
import joblib
import matplotlib
from IPython import get_ipython
from PIL import Image

### STEP 2:  LOAD THE MODEL AND ENCODER OBJECTS

#load the model and encoders
model = joblib.load("rta_model_deploy3.lzma")
encoder= joblib.load('ordinal_encoder2.joblib')

### STEP 3: SET STREAMLIT OPTIONS AND CONFIGURATIONS

st.set_page_config(page_title='Accident Severity Prediction App',  page_icon='🚧', layout='wide')

### Create OPTIONS LIST FOR THE DROP DOWN MENU

#creating options list for dropdown menu
options_day = ['Sunday', 'Monday', 'Tuesday','Wednesday', 'Thursday','Friday', 'Saturday']
options_age = [ 'Under 18','18-30', '31-50',  'Over 51', 'Unknown' ]
#number of vehicles involved was range of 1-7
#number of cosualties was range 1-8
#hour of day range of 0-23
options_types_collision= ['Collision with roadside-parked vehicles', 'Vehicle with vehicle collision',  'Collision with roadside objects', 
                          'Collision with animals', 'Other', 'Rollover', 'Fall from vehicles','Collision with pedestrians', 'With Train', 'Unknown']
options_sex = ['Male','Female','Unknown']
options_education_level = ['Elementary school','Junior high school','High school','Above high school', 'Writing & reading','Unknown', 'Illiterate' ]
options_service_year = ['Below 1yr','1-2yr','2-5yrs',  '5-10yrs','Above 10yr',  'Unknown']
options_accident_area = ['Residential areas', 'Office areas', '  Recreational areas',    ' Industrial areas', 'Other', ' Church areas', '  Market areas', 
                'Rural village areas',' Outside rural areas', ' Hospital areas', 'School areas', 'Rural village areasOffice areas' ,'Recreational areas', 'Unknown']

features = [ 'Number_of_vehicles_involved', 'Number_of_casualties','Hour_of_day','Type_of_collision','Age_band_of_driver', 'Sex_of_driver',
            'Educational_level','Service_year_of_vehicle', 'Day_of_week', 'Area_accident_occured']
#Give title to webapp using html syntax
st.markdown('Accident Severity prediction App 🚧',unsafe_allow_html=True)

#define mainfunction to take inputs fro users
def main():
    with st.form('road_traffic_severity_form'):
        st.subheader('Please Enter the following inputs:')
        no_vehicles = st.slider('Number of vehicles involved',1,7,value=0,format='%d')
        no_casualites = st.slider('Number of Casualties',1,8,value=0,format='%d')
        hour = st.slider('Hour of Day',0,23,value=0,format='%d') 
        collision = st.selectbox('Type of collision',options=options_types_collision)
        age = st.selectbox('Age group of Driver',options=options_age)
        sex = st.selectbox('Sex  of Driver',options=options_sex)
        education = st.selectbox('Education of Driver',options=options_education_level)
        service = st.selectbox('Service Year of Vehicle',options=options_service_year)
        day_of_week = st.selectbox('Day of week',options=options_day)
        area = st.selectbox('Area Accident occured',options=options_accident_area)
        
        submit = st.form_submit_button('Predict')
        
    #encode using ordinal encoder
    if submit:
        input_array = np.array([collision,age, sex,education,service,day_of_week,area],ndmin=2)
        
        encoded_arr = list(encoder.transform(input_array).ravel())
        num_arr=[no_vehicles,no_casualites,hour]
        pred_arr = np.array(num_arr+encoded_arr).reshape(1, -1)
        prediction = model.predict(pred_arr)
        if prediction == 0:
            st.write('The severity prediction is Fatal injury')
        elif prediction == 1:
            st.write('The severity prediction is Serious injury')
        else:
            st.write('The severity prediction is Slight injury')
            


a,b,c = st.columns([0.2,0.6,0.2])
with b: 
    st.image('det.jpg',width='content')


# description about the project and code files            
st.subheader("🧾Description:")
st.text("""This data set is collected from Addis Ababa Sub-city police departments for master's research work. 
The data set has been prepared from manual records of road traffic accidents of the year 2017-20. 
All the sensitive information has been excluded during data encoding and finally it has 32 features and 12316 instances of the accident.
Then it is preprocessed and for identification of major causes of the accident by analyzing it using different machine learning classification algorithms.
""")

st.markdown("Source of the dataset: [Click Here](https://www.narcis.nl/dataset/RecordID/oai%3Aeasy.dans.knaw.nl%3Aeasy-dataset%3A191591)")

st.subheader("🧭 Problem Statement:")
st.text("""The target feature is Accident_severity which is a multi-class variable. 
The task is to classify this variable based on the other 31 features step-by-step by going through each day's task. 
The metric for evaluation will be f1-score
""")


# run the main function               
if __name__ == '__main__':
   main()   

        
        
        
        
        
