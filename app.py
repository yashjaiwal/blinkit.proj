import streamlit as st
import pandas as pd
import joblib


model = joblib.load("Logistic Regression_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_col = joblib.load("columns.pkl")

st.title("Heart stroke predction by akhil❤️")
st.markdown("Provide the following details")

age = st.slider('Age',18,100,40)
sex = st.selectbox('SEX',['Male','Female'])
chest_pain = st.selectbox('Chest Pain Type',['ATA','NAP','TA','ASY'])
resting_bp = st.number_input('Resting Blood Pressure (mm Hg)',80,200,120)
cholestrol = st.number_input('Cholesterol (mg/dl)',100,600,200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl',[0,1])
resting_ecg = st.selectbox('Resting ECG', ['Normal','ST','LVH'])
max_hr = st.slider('Max Heart Rate', 60,220,150)
excercise_angina = st.selectbox('Exercise-Induced Angina',['Yes','No'])
oldpeak = st.slider('Oldpeak (ST Depression)',0.0,6.0,1.0)
st_slope = st.selectbox('ST Slope', ['Up','Flat','Down'])

if st.button('Pridict'):
    raw_input = {
        'Age' : age,
        'RestingBP':resting_bp,
        'Cholesterol':cholestrol ,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak':oldpeak,
        'Sex_' + sex:1,
        'ChestPainType_' + chest_pain :1,
        'RestingECG_' + resting_ecg:1,
        'ExerciseAngina_' + excercise_angina:1,
        'ST_Slope_' + st_slope :1

    }
    df = pd.DataFrame([raw_input])
    
    for col in expected_col:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_col]
    scaled_input = scaler.transform(df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error('⚠️⚠️High Risk of Heart Disease')
    else:
        st.success('✅✅Low Risk Of Heart Disease')    