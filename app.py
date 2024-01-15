import streamlit as st
import pickle
import numpy as np

# Full file path to 'pipe.pkl'
pipe_path = 'F:/Machine Learning Projects/laptop price prediction/pipe.pkl'

# Full file path to 'df.pkl'
df_path = 'F:/Machine Learning Projects/laptop price prediction/df.pkl'

# Load the model and dataframe
pipe = pickle.load(open(pipe_path, 'rb'))
df = pickle.load(open(df_path, 'rb'))

st.title("Laptop Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
laptop_type = st.selectbox('Type', df['TypeName'].unique())

# RAM
ram = st.selectbox('RAM in (GB)', [2, 4, 5, 8, 12, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen size
screen_size = st.number_input('Screen Size')

# Resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# CPU
cpu = st.selectbox('CPU', df['Cpu Brand'].unique())

# HDD
hdd = st.selectbox('HDD in (GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD in (GB)', [0, 128, 256, 512, 1024, 2048])

# GPU
gpu = st.selectbox('GPU', df['Gpu Brand'].unique())

# OS
os = st.selectbox('OS', df['os'].unique())

if st.button("Predict Price"):

    # Query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0
    
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # Here's the issue: Replace 'type' with 'laptop_type' to match the variable name
    query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])


    st.title("The Predicted Price of this Configuration is " + str(int(np.exp(pipe.predict(query)[0]))))
