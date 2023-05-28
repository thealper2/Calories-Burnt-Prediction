import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open("xgb.pkl", "rb"))
ss = pickle.load(open("ss.pkl", "rb"))

genders = ["Male", "Female"]

st.title("Calories Burnt Prediction")
gender = st.selectbox("Gender", genders)
age = st.number_input("Age")
height = st.number_input("Height")
heart_rate = st.number_input("Heart Rate")
body_temp = st.number_input("Body Temperature")

if st.button("Predict"):
    gender = genders.index(gender)
    test = np.array([[gender, age, height, heart_rate, body_temp]])
    test = ss.transform(test)
    res = model.predict(test)
    print(res)
    st.success("Calories Burnt: " + str(res[0]))
