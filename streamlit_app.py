import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load models
@st.cache_resource
def load_brain_model():
    return tf.keras.models.load_model(r'C:\Users\Ashish\Desktop\EEG_Epilepsy\Tumor.h5')

model_dia = pickle.load(open('diabetes_model.sav', 'rb'))
# model_dia = load_brain_model() #testing purpose
brain_model = load_brain_model()

# Sidebar navigation
with st.sidebar:
    Selected = option_menu(
        "Multiple Disease Prediction System",
        ["Diabetes Prediction", "Heart Disease Prediction", "Brain Tumour", "Parkinsons"]
    )

# Diabetes Prediction
if Selected == 'Diabetes Prediction':
    st.title("Diabetes Prediction")

    col1, col2 = st.columns(2)
    with col1:
        age = st.text_input("Age")
    with col2:
        hypertension = st.text_input("Hypertension (1 for yes / 0 for no)")
    with col1:
        heart_disease = st.text_input("Heart Disease (1 for yes / 0 for no)")
    with col2:
        bmi = st.text_input("BMI")
    with col1:
        HbA1c_level = st.text_input("HbA1c Level")
    with col2:
        blood_glucose_level = st.text_input("Glucose Level")

    res = ''
    if st.button("Diabetes Result"):
        diab_prediction = model_dia.predict([[age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level]])
        res = 'The person is Diabetic' if diab_prediction[0] == 1 else 'The person is not Diabetic'
    st.success(res)

# Brain Tumor Prediction
if Selected == 'Brain Tumour':
    st.title("Brain Tumor Detection")
    st.write("Upload an MRI image and click the 'Predict' button to see if it contains a tumor.")

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png"])
    if uploaded_file is not None:
        col1, col2 = st.columns([2, 3])
        with col1:
            st.image(uploaded_file, caption='Uploaded MRI Image', use_column_width=True)

        with col2:
            if st.button("Predict", key="prediction-result"):
                st.write("Classifying...", key="prediction-result")

                img = Image.open(uploaded_file)
                img = img.resize((150, 150))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = brain_model.predict(img_array)
                predicted_class = (prediction > 0.5).astype("int32")

                if predicted_class == 1:
                    st.markdown("<h3 style='color: green;'>Tumor detected</h3>", unsafe_allow_html=True)
                else:
                    st.markdown("<h3 style='color: red;'>No tumor detected</h3>", unsafe_allow_html=True)

# Heart Disease Prediction
if Selected == 'Heart Disease Prediction':
    st.title("Heart Disease Prediction")
    st.write("Feature under development.")


if (Selected ==  'Parkinsons'):
    
    st.title("Parkinsons Prediction:")

    col1,col2,col3=st.columns(3)
    
    with col1:
        MDVPFo_Hz= st.text_input("MDVP:Fo(Hz)")
    with col2:
        hMDVPFhi_Hz= st.text_input("MDVP:Fhi(Hz)")
    with col3:
        hMDVPFlo_Hz=st.text_input("MDVP:Flo(Hz)")
    with col1:
        MDVPJitter_Abs=st.text_input("MDVP:Jitter(Abs)")
    with col2:
        MDVp_RAP=st.text_input("MDVP:RAP")
    with col3:
        MDVp_PPQ=st.text_input("MDVP:PPQ")
    with col1:
        Jitter_DDP= st.text_input("Jitter:DDP")
    with col2:
        MDVP_Shimmer= st.text_input("MDVP:Shimmer")
    with col3:
        MDVP_ShimmerdB=st.text_input("MDVP:Shimmer(dB)")
    with col1:
        Shimmer_APQ3=st.text_input("Shimmer:APQ3")
    with col2:
        Shimmer_APQ5=st.text_input("Shimmer:APQ5")
    with col3:
        MDVp_APQ=st.text_input("MDVP:APQ")
    with col1:
        Shimmer_DDA= st.text_input("Shimmer:DDA")
    with col2:
        NHr= st.text_input("NHR")
    with col3:
        HNr= st.text_input("HNR")
    with col1:
        RPDe=st.text_input("RPDE")
    with col2:
        DFa=st.text_input("DFA")
    
    with col3:
        spread1=st.text_input("spread1")
    with col1:
        spread2=st.text_input("spread2")
    with col2:
        d2=st.text_input("D2")
    with col3: 
        PPe=st.text_input("PPE")    

#result 
    res4=''
    
#button
    if st.button("Parkinsons Result"):
        park_prediction =model_park.predict([[MDVPFo_Hz,hMDVPFhi_Hz,hMDVPFlo_Hz,MDVPJitter_Abs,MDVp_RAP,MDVp_PPQ,Jitter_DDP,MDVP_Shimmer,MDVP_ShimmerdB,Shimmer_APQ3,Shimmer_APQ5,MDVp_APQ,Shimmer_DDA,NHr,HNr,RPDe,DFa,spread1,spread2,d2,PPe]])
        
        if park_prediction[0]==0:
            res4='The Person does not have Parkinsons'
        else:
            res4="The Person has Parkinsons"
    
    st.success(res4)
