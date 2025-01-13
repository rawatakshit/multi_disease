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
    return tf.keras.models.load_model(r'Tumor.h5')

model_dia = pickle.load(open('diabetes_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))
model_park = pickle.load(open('parkinsons_model.sav','rb'))
model_heart=pickle.load(open("heart_disease_model.sav",'rb'))
brain_model = load_brain_model()

# Sidebar navigation
with st.sidebar:
    Selected = option_menu(
        "Multiple Disease Prediction System",
        ["Diabetes Prediction","Brain Tumour", "Parkinsons","Heart Disease"]
    )


# Diabetes Prediction
if Selected == 'Diabetes Prediction':

    st.title("Diabetes Prediction")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.text_input('Gender (1 for Male, 0 for Female, 2 for Other)')
        hypertension = st.text_input("Hypertension (1 for Yes, 0 for No)")
        bmi = st.text_input("BMI")
        blood_glucose_level = st.text_input("Blood Glucose Level")
    with col2:
        age = st.text_input("Age")
        heart_disease = st.text_input("Heart Disease (1 for Yes, 0 for No)")
        HbA1c_level = st.text_input("HbA1c Level")

     # Result
    res = ''

    # Prediction function
   error_flag = False
try:
    gender = int(gender)
    age = float(age)
    hypertension = int(hypertension)
    heart_disease = int(heart_disease)
    bmi = float(bmi)
    hba1c = float(hba1c)
    glucose = float(glucose)

    # Check for valid ranges
    if gender not in [0, 1, 2]:
        st.error("Invalid input for gender. Enter 0 (Female), 1 (Male), or 2 (Other).")
        error_flag = True
    if age < 0 or age > 120:
        st.error("Invalid input for age. Enter a value between 0 and 120.")
        error_flag = True
    if hypertension not in [0, 1]:
        st.error("Invalid input for hypertension. Enter 0 (No) or 1 (Yes).")
        error_flag = True
    if heart_disease not in [0, 1]:
        st.error("Invalid input for heart disease. Enter 0 (No) or 1 (Yes).")
        error_flag = True
    if bmi <= 0:
        st.error("Invalid input for BMI. Enter a positive value.")
        error_flag = True
    if hba1c <= 0:
        st.error("Invalid input for HbA1c Level. Enter a positive value.")
        error_flag = True
    if glucose <= 0:
        st.error("Invalid input for glucose level. Enter a positive value.")
        error_flag = True

except ValueError:
    st.error("Please enter valid numeric values for all fields.")
    error_flag = True

# Prediction button and result
if st.button("Predict Diabetes"):
    if not error_flag:
        # Prepare input array
        input_data = np.array([gender, age, hypertension, heart_disease, bmi, hba1c, glucose])
        input_data_reshaped = input_data.reshape(1, -1)

        # Scale the input data
        scaled_input = scaler.transform(input_data_reshaped)

        # Make prediction
        prediction = model.predict(scaled_input)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        st.success(f"The patient is: {result}")
    else:
        st.error("Please correct the errors above before making a prediction.")
    
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


#  Parkinsons Prediction
if (Selected ==  'Parkinsons'):
    
    st.title("Parkinsons Prediction:")

    col1,col2,col3=st.columns(3)
    
    with col1:
        MDVPFo_Hz= st.text_input("MDVP:Fo(Hz)")
        MDVPJitter_Abs=st.text_input("MDVP:Jitter(Abs)")
        Jitter_DDP= st.text_input("Jitter:DDP")
        Shimmer_APQ3=st.text_input("Shimmer:APQ3")
        Shimmer_DDA= st.text_input("Shimmer:DDA")
        RPDe=st.text_input("RPDE")
        spread2=st.text_input("spread2")
    with col2:
        hMDVPFhi_Hz= st.text_input("MDVP:Fhi(Hz)")
        MDVp_RAP=st.text_input("MDVP:RAP")
        MDVP_Shimmer= st.text_input("MDVP:Shimmer")
        Shimmer_APQ5=st.text_input("Shimmer:APQ5")
        NHr= st.text_input("NHR")
        DFa=st.text_input("DFA")
        d2=st.text_input("D2")

    with col3:
        hMDVPFlo_Hz=st.text_input("MDVP:Flo(Hz)")
        MDVp_PPQ=st.text_input("MDVP:PPQ")
        MDVP_ShimmerdB=st.text_input("MDVP:Shimmer(dB)")
        MDVp_APQ=st.text_input("MDVP:APQ")
        HNr= st.text_input("HNR")
        spread1=st.text_input("spread1")
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


# Heart Disease Prediction in Streamlit
if Selected == 'Heart Disease':
    st.title("Heart Disease Prediction")
    col1, col2 = st.columns(2)

    # Collect inputs
    with col1:
        age = st.text_input('Age', key="heart_age")
        cp = st.text_input("Chest Pain Type (cp)")
        chol = st.text_input("Cholesterol (chol)")
        restecg = st.text_input("Resting ECG (restecg)")
        exang = st.text_input("Exercise Induced Angina (exang)")
        slope = st.text_input("Slope of ST Segment (slope)")
        thal = st.text_input("Thalassemia (thal)")
    with col2:
        sex = st.text_input("Gender (1 for Male, 0 for Female)",key="heart_sex")
        trestbps = st.text_input("Resting Blood Pressure (trestbps)")
        fbs = st.text_input("Fasting Blood Sugar (fbs)")
        thalach = st.text_input("Maximum Heart Rate Achieved (thalach)")
        oldpeak = st.text_input("ST Depression (oldpeak)")
        ca = st.text_input("Number of Major Vessels (ca)")
    

    # Result
    result = ''

    # Function to predict heart disease
    def heart_prediction(data):
        # Convert input to numpy array
        array = np.asarray(data, dtype=float)  # Ensure all data is numeric
        data_reshape = array.reshape(1, -1)
        prediction = model_heart.predict(data_reshape)
        if prediction[0] == 0:
                 return "The Person does not have Heart Disease"
        else:
                return "The Person has Heart Disease"

    # Button for prediction
    if st.button("Heart Disease Result"):
        # Validate and convert inputs to float
        try:
            inputs = [
                float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs),
                float(restecg), float(thalach), float(exang), float(oldpeak), float(slope),
                float(ca), float(thal)
            ]
            result = heart_prediction(inputs)
        except ValueError:
            result = "Please enter valid numeric values for all fields."

        st.success(result)




