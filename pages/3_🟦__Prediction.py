# import required dependencies
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import lime
import lime.lime_tabular

# load the trained models
with open('models/lr.pkl', 'rb') as f:
    lr = pickle.load(f)
with open('models/tree.pkl', 'rb') as f:
    tree = pickle.load(f)
with open('models/rf.pkl', 'rb') as f:
    rf = pickle.load(f)
with open('models/svc.pkl', 'rb') as f:
    svc = pickle.load(f)
with open('models/gbc.pkl', 'rb') as f:
    gbc = pickle.load(f)

model_option = {'Logistic Regression':lr,
                'Decision Tree': tree,
                'Random Forest': rf,
                'Support Vector Machine': svc,
                'Gradient Boosting': gbc}

# load the preprocess encoder and scaler
with open('preprocess/astage_coder.pkl', 'rb') as f:
    astage_coder = pickle.load(f)
with open('preprocess/estrogen_coder.pkl', 'rb') as f:
    estrogen_coder = pickle.load(f)
with open('preprocess/marital_coder.pkl', 'rb') as f:
    marital_coder = pickle.load(f)
with open('preprocess/ordinal_coder.pkl', 'rb') as f:
    ordinal_coder = pickle.load(f)
with open('preprocess/progesterone_coder.pkl', 'rb') as f:
    progesterone_coder = pickle.load(f)
with open('preprocess/race_coder.pkl', 'rb') as f:
    race_coder = pickle.load(f)
with open('preprocess/robust_scaler.pkl', 'rb') as f:
    robust_scaler = pickle.load(f)
with open('preprocess/target_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# load the dataset
training = pd.read_csv('data/training_data.csv')
Xtrain = training.drop(columns=['Status'], axis=1)
Xtrain_array = Xtrain.to_numpy()
ytrain = training['Status']


# Streamlit page configuration
st.set_page_config(
    page_title='Prediction',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title(':blue[Survival Prediction]')

#######################################################################################################################################
# container - User Information Input
container = st.container(border=True)
container.markdown('### :blue[Please Provide the Information for Prediction]')
container.markdown('### ')
container.markdown('#### Demographic Information')
col1, col2, col3 = container.columns(3)
Age = col1.number_input('Enter your age (years):', value=21)
Race = col2.selectbox('Select your race', ['Other (American Indian/AK Native, Asian/Pacific Islander)','White', 'Black'])
Marital_Status = col3.selectbox('Select your marital status', ['Married (including common law)', 'Divorced', 'Single (never married)', 'Widowed', 'Separated'])
container.markdown('### ')

container.markdown('#### Medical Condition')
col4, col5, col6, = container.columns([1,1,4])
Progesterone_Status = col4.selectbox('Select the progesterone status', ['Positive', 'Negative'])
Estrogen_Status = col5.selectbox('Select the estrogen status', ['Positive', 'Negative'])
Grade = col6.selectbox('Select the grading and differentiation codes', ['Well differentiated; Grade I', 'Moderately differentiated; Grade II', 'Poorly differentiated; Grade III', 'Undifferentiated; anaplastic; Grade IV'])

col7, col8, col9, col10 = container.columns(4)
A_Stage = col7.selectbox('Select the historic stage A', ['Regional', 'Distant'])
N_Stage = col8.selectbox('Select the N stage (involvement of nearby lymph nodes)', ['N1','N2','N3'])
T_Stage = col9.selectbox('Select the T Stage (size and extent of the primary tumor)', ['T1','T2','T3','T4'])
Sixth_Stage = col10.selectbox('Select the Breast Adjusted AJCC 6th Stage', ['IIA','IIB','IIIA','IIIB','IIIC'])

col11, col12 = container.columns(2)
Tumor_Size = col11.number_input('Enter the tumor size', value=15.00)
Survival_Month = col12.number_input('Enter the survival months', value=5)

col13, col14 = container.columns(2)
Regional_Node_Examined = col13.number_input('Enter the total number of regional lymph nodes that were removed and examined by the pathologist', value=2)
Regional_Node_Positive = col14.number_input('Enter the exact number of regional lymph nodes examined by the pathologist that were found to contain metastases', value=2)


user_ordinal_feats = [[T_Stage, N_Stage, Sixth_Stage, Grade]]
user_progesterone = [[Progesterone_Status]]
user_estrogen = [[Estrogen_Status]]
user_race = [[Race]]
user_marital = [[Marital_Status]]
user_astage = [[A_Stage]]
user_numerical = np.array([Age, Tumor_Size, Regional_Node_Examined, Regional_Node_Positive, Survival_Month])
#######################################################################################################################################



#######################################################################################################################################
# container 2 - Prediction Results
container2 = st.container(border=True)
container2.markdown('### :blue[Prediction Result]')
container2.markdown('### ')
model = container2.selectbox('Select the model for prediction:', model_option.keys())
submit = container2.button('Predict', type='primary')
container2.write(''' :red[Disclaimer: Please note that the predictions generated by our application are based on statistical models and 
should not replace professional medical advice. Always consult with a healthcare professional for accurate diagnosis and 
personalized recommendations.]''')

if submit:
    user_ordinal_encoded = ordinal_coder.transform(user_ordinal_feats)
    user_progesterone_encoded = progesterone_coder.transform(user_progesterone)
    user_estrogen_encoded = estrogen_coder.transform(user_estrogen)
    user_race_encoded = race_coder.transform(user_race)
    user_marital_encoded = marital_coder.transform(user_marital)
    user_astage_encoded = astage_coder.transform(user_astage)

    user_array = np.hstack((user_ordinal_encoded, user_progesterone_encoded.reshape(1, 1)))
    user_array = np.hstack((user_array, user_estrogen_encoded.reshape(1,1)))
    user_array = np.hstack((user_array, user_race_encoded.toarray()))
    user_array = np.hstack((user_array, user_marital_encoded.toarray()))
    user_array = np.hstack((user_array, user_astage_encoded.toarray()))
    user_array = np.hstack((user_array, user_numerical.reshape(1,-1)))

    user_input_scaled = robust_scaler.transform(user_array)

    prediction = model_option[model].predict(user_input_scaled)

    #outcome = lr.predict(user_input_scaled)
    final = label_encoder.inverse_transform(prediction)

    if final[0] == 'Alive':
        container2.subheader('Based on the model prediction, the patient has a :blue[higher chance] to alive.')
    else:
        container2.subheader('Based on the model prediction, the patient has a :red[lower chance] to alive.')

    container3 = st.container(border=True)
    container3.markdown('### :blue[Model Interpretability]')

    model_explainer = lime.lime_tabular.LimeTabularExplainer(Xtrain_array, feature_names=Xtrain.columns, class_names=['Alive','Dead'], mode='classification')
    exp = model_explainer.explain_instance(user_input_scaled.reshape(1,-1)[0], model_option[model].predict_proba, num_features=21)
    # fig = exp.as_pyplot_figure()
    # container2.pyplot(fig)

    res = exp.as_list()
    res_table = pd.DataFrame(res, columns=['Predictor','Score'])
    styled_res_table = res_table.style.set_table_styles([
    {
        'selector': 'thead th',
        'props': [('background-color', '#57a4fe'),  
                  ('color', 'black')]
    },
    {
        'selector': 'tr:hover',
        'props': [('background-color', '#cff5ff'),
                   ('color', 'black')]
    }],
    overwrite=False
    ).set_properties(**{'padding': '10px'})


    barchart = px.bar(res_table.iloc[::-1].head(21), x='Score', y='Predictor', height=800, title='Local Explanation for Class Dead')
    barchart.update_layout(title_font=dict(size=25))
    barchart.update_layout(template='plotly_dark')
    container3.plotly_chart(barchart, theme='streamlit', use_container_width=True)

    container3.table(styled_res_table)
#######################################################################################################################################






