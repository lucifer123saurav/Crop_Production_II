# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:49:46 2023

@author: hc
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle 
# st.set_page_config(page_title="Crop Production", page_icon=":corn:")


st.set_page_config(layout="wide")

video_html = """
		<style>
		#myVideo {
		  position: fixed;
		  right: 0;
		  bottom: 0;
		  min-width: 100%; 
		  min-height: 100%;
		}
		.content {
		  position: fixed;
		  bottom: 0;
		  background: rgba(0, 0, 0, 0.5);
		  color: #f1f1f1;
		  width: 100%;
		  padding: 20px;
		}
		</style>	
		<video autoplay muted loop id="myVideo">
		  <source src="https://v1.pinimg.com/videos/mc/720p/8b/0d/ef/8b0def62d7fa9da5f1734c1e4e23294e.mp4")>
		  Your browser does not support HTML5 video.
		</video>
        """
st.markdown(video_html, unsafe_allow_html=True)

st.header ('Predict the Production of Crops at any Particular Season')

with open('mapping_dict.pkl','rb')as f:
    mapping_dict=pickle.load(f)

with open('model.pkl','rb')as f:
    model=pickle.load(f)

with open('original_data.pkl','rb')as f:
     data = pd.read_pickle(f)

def predict(State,District,Crop,Season,Area):
    state=mapping_dict['State'][State]
    district=mapping_dict['District'][District]
    crop=mapping_dict['Crop'][Crop]
    season=mapping_dict['Season'][Season]


    prediction=model.predict(pd.DataFrame(np.array([state,district,crop,season,Area]).reshape(1,5),columns=['State','District','Crop','Season','Area']))
    return prediction

# input 
#st.image("C:\Users\hc\Desktop\new_crop\pexels-pixabay-531880.jpg")
state_list =data['State'].unique()
selected_state=st.selectbox(
    "Select a state from the Dropdown",
    options=state_list
)

district_list =data['District'].unique()
selected_district=st.selectbox(
    "Select a district from the Dropdown",
    options=district_list
)
season_list=data['Season'].unique()
selected_season=st.selectbox(
    "Select a district from the Dropdown",
    options=season_list
)

crop_list =data['Crop'].unique()
selected_crop=st.selectbox(
    "Select a state from the Dropdown",
    options=crop_list
)
Area = st.number_input('Area of plot in(Hectares)',min_value=0.00001,max_value=100000000.0,value=1.0)

if st.button('Predict Production'):
    st.subheader(predict(selected_state,selected_district,selected_crop,selected_season,Area))
    st.subheader('Tonnes')

css = """
h1 {
    color: blue;
    font-size: 36px;
    text-align: center;
}
p {
    color: white;
    font-size: 24px;
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}
"""

# Render the CSS styles using st.markdown
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)