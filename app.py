import streamlit as st
import pickle
import numpy as np
import pandas as pd

df = pickle.load(open('df.pkl', 'rb'))
model = pickle.load(open('model4.pkl', 'rb'))

st.title('Laptop Price Predictor')

company = st.selectbox('Brand', df['Company'].unique())
type_name = st.selectbox('Type', df['TypeName'].unique())
screen_size = st.number_input('Screen Size')
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
cpu = st.selectbox('CPU', df['CPU'].unique())
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu_brand'].unique())
os = st.selectbox('OS', df['OS'].unique())

if st.button('Predict Price'):
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    dff = {'Company': company, 'TypeName': type_name, 'Inches': screen_size, 'Ram': ram,
           'Weight': weight, 'Touchscreen': touchscreen, 'Ips': ips, 'CPU': cpu, 'HDD': hdd,
           'SSD': ssd, 'Gpu_brand': gpu, 'OS': os}

    df_new = df.drop('Price', axis=1).append(dff, ignore_index=True)
    categorical_cols = ['Company', 'TypeName', 'CPU', 'Gpu_brand', 'OS']
    df_new = pd.get_dummies(df_new, columns=categorical_cols)
    query = df_new.iloc[-1].values
    query = np.reshape(query, (-1, 43))
    st.title("The predicted price of this configuration is " +
             str(int(np.exp(model.predict(query)[0]))))
