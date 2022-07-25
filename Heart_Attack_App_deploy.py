# -*- coding: utf-8 -*-
"""
"""

# From the model, there are 7 selected features include 
# age,thalachh,oldpeak,cp,exng,caa, and thall
#%% App Deployment
import os
import pickle
import numpy as np
import streamlit as st
from PIL import Image

MODEL_PATH = os.path.join(os.getcwd(),'Models','HAP_App_model.pkl')
with open(MODEL_PATH,'rb') as file:
    classifier = pickle.load(file)

st.markdown("<h1 style='text-align: center; color: black;'> Heart Attack App </h1>",
            unsafe_allow_html=True)

# Main Page
st.header('What is Heart Attack?')
st.video('https://youtu.be/bw_Vv2WRG-A')

# SideBar Page
st.sidebar.header("Please fill in the details below")

age =       st.sidebar.number_input("Age in Years", 1, 90, 25, 1)
thalachh = st.sidebar.number_input("Maximum Heart Rate Achieved",0,200,150,1)
oldpeak = st.sidebar.number_input("ST depression induced by exercise" 
                                  "relative to rest",
                                  0.00, 3.50, 2.3, 0.10)
cp = st.sidebar.number_input("Chest Pain type",0,3,1,1)
exng = st.sidebar.number_input("Exercise Induced Angina",0,1,1,1)
caa = st.sidebar.number_input("Number of major vessels (0-3)",0,3,1,1)
thall = st.sidebar.number_input("Thalassemia",0,3,1,1)

# Every form must have a submit button.
submitted = st.sidebar.button("Submit")
st.markdown(
            "<h1 style='text-align: center; color: black;'>Your Result</h1>",
            unsafe_allow_html=True)
if submitted:
    new_data = np.expand_dims([age,thalachh,oldpeak,cp,exng,caa,thall],
                                  axis=0)
    outcome = classifier.predict(new_data)[0]
    if outcome == 0:
            st.markdown(
            "<h3 style='text-align: center; color: black;'>Congrats you are healthy, no risk of heart attack, keep it up!!</h3>",
            unsafe_allow_html=True)
            st.markdown(
                "![Alt Text](https://media.giphy.com/media/kwDQ9I0oAPVqE/giphy.gif)")
            st.balloons()
    else:
        st.markdown(
                "<h3 style='text-align: center; color: black;'>You have a risk of experiencing a <b>heart attack</b>. Consult a doctor immediately</h3>",
                        unsafe_allow_html=True)
        st.image(Image.open(os.path.join(os.getcwd(),'Statics','HA_tips.png')))
        st.snow()
