# import required dependencies
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pywaffle import Waffle
import plotly.express as px
import helper

# Streamlit page configuration
st.set_page_config(
    page_title='Breast Cancer Survivorship Prediction',
    page_icon=':desktop_computer:',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title('Breast Cancer Survivorship Prediction')
st.image('image/cover_img2.png', use_column_width=True)
st.markdown('### ')

#################################################################################################################################
# container 2 - facts on statistics and trends
container2 = st.container(border=True)
container2.markdown('### :blue[Statistics and Trends]')
col1, col2 = container2.columns(2)
col1.image('image/breast_cancer.jpg')
col2.subheader('''According to the World Health Organization (WHO), there were approximately 2.3 million new cases of breast cancer in 2020, representing about 11.7% of all cancer cases globally.''')
col2.markdown('Reference:')
col2.markdown('''
Sung, H., Ferlay, J., Siegel, R. L., Laversanne, M., Soerjomataram, I., Jemal, A., & Bray, F. (2021). 
Global Cancer Statistics 2020: GLOBOCAN Estimates of Incidence and Mortality Worldwide for 36 Cancers in 185 Countries. 
<i>CA: a cancer journal for clinicians</i>, 71(3), 209–249. https://doi.org/10.3322/caac.21660''', unsafe_allow_html=True)

col3, col4 = container2.columns(2)
col3.image('image/breast_cancer3.jpg')
col4.subheader('Breast cancer was the most common cancer in women in 157 countries out of 185 in 2022.')
col4.markdown('Reference')
col4.markdown('''
Sung, H., Ferlay, J., Siegel, R. L., Laversanne, M., Soerjomataram, I., Jemal, A., & Bray, F. (2021). 
Global Cancer Statistics 2020: GLOBOCAN Estimates of Incidence and Mortality Worldwide for 36 Cancers in 185 Countries. 
<i>CA: a cancer journal for clinicians</i>, 71(3), 209–249. https://doi.org/10.3322/caac.21660''', unsafe_allow_html=True)

col5, col6 = container2.columns(2)
col5.image('image/breast_cancer2.jpg')
col6.subheader('Women in countries with a lower Human Development Index (HDI) are 50% less likely to be diagnosed with breast cancer compared to those in high HDI countries; however, they face a significantly higher risk of dying from the disease due to late diagnosis and insufficient access to quality treatment.')
col6.markdown('Reference')
col6.markdown('''
Sung, H., Ferlay, J., Siegel, R. L., Laversanne, M., Soerjomataram, I., Jemal, A., & Bray, F. (2021). 
Global Cancer Statistics 2020: GLOBOCAN Estimates of Incidence and Mortality Worldwide for 36 Cancers in 185 Countries. 
<i>CA: a cancer journal for clinicians</i>, 71(3), 209–249. https://doi.org/10.3322/caac.21660''', unsafe_allow_html=True)

col7, col8 = container2.columns(2)
col7.image('image/breast_cancer4.PNG')
col8.subheader('Non-Hispanic white women and non-Hispanic Black women have the highest incidence of breast cancer (rate of new breast cancer cases) overall. Hispanic women have the lowest incidence.''')
col8.markdown('Reference')
col8.markdown('''
Surveillance Research Program, National Cancer Institute. SEER*Explorer. 
Breast Cancer – 5-year age-adjusted incidence rates, 2016-2020, by race/ethnicity, female, all ages, all stages. 
https://seer.cancer.gov/explorer/, 2023.''', unsafe_allow_html=True)
###################################################################################################################################


###################################################################################################################################
# container 3 - Objective of the Project
container3 = st.container(border=True)
container3.markdown('### :blue[Objective of the Project]')
container3.write('''
The primary objective of this project is to develop a user-friendly application that healthcare providers can use to predict and monitor the survival outcomes of breast cancer patients. The application is designed with the following goals in mind:
1. Accessibility: Create an intuitive interface that healthcare providers can easily navigate.
2. Clinical Utility: Offer a tool that supports clinical decision-making by providing timely and relevant prognostic information.
''')
###################################################################################################################################



###################################################################################################################################
# container 4 - Navigation on the Application
container4 = st.container(border=True)
container4.markdown('### :blue[How to use this Application?]')
container4.write('''
Our web application consists of four pages:
1. Home: Provides background information about the project.
2. About the Dataset: Allows you to explore and understand the SEER Breast Cancer dataset used for training the model.
3. Model Performance: Displays the performance metrics and evaluation results of the trained machine learning model.
4. Prediction: Lets you input new patient data and obtain survival predictions from the model.
''')
####################################################################################################################################