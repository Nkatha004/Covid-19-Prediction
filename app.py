import joblib
import streamlit as st
from PIL import Image

classifier = joblib.load("RandomForest.pkl")

def prediction_generator(cough, fever, sore_throat, shortness_of_breath,
                         head_ache,gender, age_60, test_ind, month, day):
    prediction = classifier.predict([[cough, fever, sore_throat, shortness_of_breath, head_ache,
                                      gender,age_60, test_ind, month, day]])

    if prediction == 0:
        output = 'Negative'
    else:
        output = 'Positive'

    return output

def main():
    image = Image.open("logo.jpg")
    st.image(image, width = 500)

    month = st.selectbox('Month', (i for i in range(1,13)))
    day = st.selectbox('Day \ Date', (i for i in range(1,32)))

    gender = st.selectbox('Gender',('Female', 'Male'))
    if gender == 'Female':
        gender = 0
    else:
        gender = 1

    st.text("Symptoms:")
    cough = st.checkbox('Cough')
    fever = st.checkbox('Fever')
    sorethroat = st.checkbox('Sore Throat')
    headache = st.checkbox('Headache')
    shortness_of_breath = st.checkbox('Shortness of breath')

    age_60 = st.selectbox('Are you 60 years and above?', ('No', 'Yes'))

    if age_60 == 'Yes':
        age_60 = 1
    else:
        age_60 = 0

    test_ind = st.selectbox('What is the primary reason for predicting your health state? ',
                                ('I have been in contact with confirmed case', 'I was abroad', 'Other'))

    if test_ind == 'I have been in contact with confirmed case':
        test_ind = 0
    elif test_ind == 'I was abroad':
        test_ind = 1
    else:
        test_ind = 2


    if st.button('Predict'):
        output = prediction_generator(cough, fever, sorethroat, shortness_of_breath, headache,gender,age_60,
                                      test_ind, month, day)
        st.success(f"Prediction: {output} for the novel corona virus")


if __name__ == '__main__':
    main()
