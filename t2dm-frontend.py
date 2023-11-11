import streamlit as st

# Streamlit page configuration - must be at the top
st.set_page_config(page_title='Diabetes Prediction from Voice', layout='wide')


import pandas as pd
import feat_extract
import ml_analysis
import numpy as np
import os


# Page header
st.title('Diabetes Prediction from Voice Analysis')
st.write('Upload a voice file to analyze and predict the likelihood of Type 2 Diabetes Mellitus.')

# Example text for users to read
st.markdown("""
### Suggested Reading Passage
Please read the following passage aloud and record your voice. This passage is designed to capture a wide range of vocal characteristics.

_The Rainbow Passage:_
When the sunlight strikes raindrops in the air, they act like a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end. People look, but no one ever finds it.
""")

# File upload
uploaded_file = st.file_uploader("Choose a voice file...", type=['wav'])

def single_prediction(features_df, model_type='svm'):
    # Simulating a single data frame for prediction
    df = pd.concat([features_df]*2)  # Duplicating the row to mimic train-test split
    df['ID'] = [1, 2]  # Dummy IDs
    df['Diagnosis'] = [0, 1]  # Dummy diagnosis labels

    # Features used in the model
    feats = list(features_df.columns)

    # Perform prediction
    test_results = ml_analysis.pipeline_fun(df, feats, [1], [2], [1], [2], 1, model_type)
    return test_results.iloc[0]['Fold 1 Probability']

if uploaded_file is not None:
    # Save the uploaded voice file
    with open(os.path.join("uploaded_files", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success('File Uploaded Successfully!')

    # Extract features from the voice file
    features = feat_extract.measure_feats(os.path.join("uploaded_files", uploaded_file.name), f0min=75, f0max=300, unit="Hertz")

    # Convert features to a pandas DataFrame
    features_df = pd.DataFrame([features], columns=feat_extract.feature_names)

    # Predict using the machine learning model
    prediction_probability = single_prediction(features_df)

    # Display the prediction
    if prediction_probability > 0.5:  # Assuming a threshold of 0.5 for diabetes prediction
        st.warning('The model predicts a high likelihood of Type 2 Diabetes Mellitus.')
    else:
        st.success('The model predicts a low likelihood of Type 2 Diabetes Mellitus.')
else:
    st.info('Please upload a voice file to proceed.')

# Study results and disclaimer expander
with st.expander("Study Summary and Disclaimer"):
    st.markdown("""
    [Acoustic Analysis and Prediction of Type 2 Diabetes Mellitus Using Smartphone-Recorded Voice Segments](https://www.mcpdigitalhealth.org/article/S2949-7612(23)00073-1/fulltext)
    
    The study reported an optimal accuracy of 0.75±0.22, with a specificity of 0.77±0.29 and a sensitivity of 0.73±0.23 from cross-validation of the matched dataset. When predicting the test set, the accuracy was higher at 0.89, with a specificity of 0.91 and sensitivity of 0.71.

    The accuracy of this app cannot be assumed to be the same as the study without verification. The length of the recording may affect performance; a longer recording may improve it, or it may lead to "model fatigue".

    Always consult with a healthcare professional.
    """)

# About section
st.sidebar.header('About')
st.sidebar.info('This application uses machine learning to analyze voice recordings and predict the likelihood of Type 2 Diabetes Mellitus. The analysis is based on acoustic features extracted from the voice recording.')
