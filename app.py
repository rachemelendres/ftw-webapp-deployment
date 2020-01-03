import pandas as pd
import streamlit as st
import joblib
import numpy as np
import time
from PIL import Image

# Title of your Web Application
# st.title('Sales Forecasting')

# # Describe your Web Application
# # st.write('We demonstrate how we can forecast advertising sales on an ad expenditure')

### Writing a title and description using magic commands
"""
# Sales Forecasting Demo App
This is a demo application created to demonstrate how we can forecast advertising sales on an ad expenditure. 
"""
### Read Data
data = pd.read_csv('data/advertising_regression.csv')

image = Image.open('sales.jpg')
st.image(image, caption='Image grabbed from https://www.saleshacker.com/how-to-forecast-sales/', use_column_width=True)                  
    
### Sidebar widgets 

# Create sidebar
st.sidebar.subheader('Advertising Costs Slider')
st.sidebar.markdown('Toggle slider to try different advertising costs for TV, Radio, and Newspaper')

# TV Slider
TV = st.sidebar.slider('TV Advertising Cost', 0, 300, 150)

# Radio Slider
radio = st.sidebar.slider('Radio Advertising Cost', 0, 50, 25)

# Newspaper Slider
newspaper = st.sidebar.slider('Newspaper Advertising Cost', 0, 250, 125)



### Prediction sales generator

# Load saved ML Model
saved_model = joblib.load('advertising_model.sav')

# Predict sales using variables/features
predicted_sales = saved_model.predict([[TV, radio, newspaper]])[0]

# Print predictions
st.subheader('Prediction Sales Generator')
st.write('Input desired costs for TV, Radio, and Newspaper using interactive slider on the left to generate predicted sales.')
st.write(f"**_Predicted sales is {predicted_sales} dollars._**")

### Advertising data options for user

st.subheader('Advertising Data')
st.write('Click for more information on Advertising Data')

# Show Data
if st.button('Load Advertising data'):
    'Loading data..'
    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
  # Update the progress bar with each iteration.
        latest_iteration.text(f'Progress {i+1}%')
        bar.progress(i + 1)
        time.sleep(0.05)

    '...and now we\'re done!'
    
    st.subheader('Advertising Raw Data')
    st.write('Data obtained from an original dataset created by Jose Portilla and Pierian Data for his Udemy Course (Python for Data Science and Machine Learning Bootcamp)') 
    st.write(data)
    
### Histogram charts   

if st.button('Explore my data'):
    'Loading histogram charts..'
    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
  # Update the progress bar with each iteration.
        latest_iteration.text(f'Progress {i+1}%')
        bar.progress(i + 1)
        time.sleep(0.05)

    '...and now we\'re done!'
    st.subheader('Radio Advertising Cost Distribution')

    # Radio Histogram
    hist_values = np.histogram(data.radio, bins=300, range=(0,300))[0]

    # Show Bar Chart
    st.bar_chart(hist_values)

    st.subheader('Newspaper Advertising Cost Distribution')

    # Newspaper Histogram
    hist_values = np.histogram(data.newspaper, bins=300, range=(0,300))[0]

    # Show Bar Chart
    st.bar_chart(hist_values)

    st.subheader('TV Advertising Cost Distribution')

    # TV Histogram
    hist_values = np.histogram(data.TV, bins=300, range=(0,300))[0]

    # Show Bar Chart
    st.bar_chart(hist_values)


