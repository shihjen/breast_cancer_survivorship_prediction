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
    page_title='About the Dataset',
    page_icon=':chart_with_upwards_trend:',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title(':blue[About the Dataset]')


# import dataset
@st.cache_data
def get_data():
    data = pd.read_csv('data/seer_breastcancer_dataset.csv')
    summary_stat = pd.read_csv('data/summary_stat.csv')
    groups = data.groupby('Status')
    alive = groups.get_group('Alive')
    dead = groups.get_group('Dead')
    return data, summary_stat, alive, dead

data, summary_stat, alive, dead = get_data()
data_option = {'All': data, 'Alive': alive, 'Dead': dead}

def plot_scatter(data, var1, var2, container):
    import plotly.express as px
    custom_colors = ['#57a4fe', '#cff5ff']
    if var1 == var2:
        title = f'Distribution of {var1}'
    else:
        title = f'Correlation between {var1} and {var2}'
    scatterplot = px.scatter(data, x=var1, y=var2, color='Status', height=700, color_discrete_sequence=custom_colors, title=title)
    scatterplot.update_layout(title_font=dict(size=25), template='plotly_dark')
    container.plotly_chart(scatterplot, theme='streamlit', use_container_width=True)

###############################################################################################################################
# container --- dataset information
container = st.container(border=True)
categorical_var = ['Race', 'Marital Status', 'T Stage', 'N Stage', '6th Stage', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status', 'Status']
container.markdown('### :blue[SEER Breast Cancer Dataset]')
container.write('''
This dataset of breast cancer patients was sourced from the November 2017 update of the SEER Program of the NCI, 
which provides population-based cancer statistics. 
It includes female patients diagnosed with infiltrating duct and lobular carcinoma of the breast (SEER primary site recode NOS histology codes 8522/3) 
between 2006 and 2010. Patients with unknown tumor size, unknown examined regional lymph nodes, unknown regional positive lymph nodes, 
or a survival time of less than one month were excluded, resulting in a final cohort of 4,024 patients.
''')
container.markdown('#### :blue[Dataset Source]')
container.markdown('''JING TENG. (2019). [SEER Breast Cancer Data](https://ieee-dataport.org/open-access/seer-breast-cancer-data). IEEE Dataport.''', unsafe_allow_html=True)

container.markdown('#### :blue[Variable Description]')
container.write('''
1. Age: The age of the patient at diagnosis.
2. Race: Race recode is based on the race variables and the American Indian/Native American IHS link variable. 
This recode should be used to link to the populations for white, black and other.
3. Marital Status: Thepatient\'s marital status at the time of diagnosis.
4. T Stage: Part of the TNM staging system, T stage refers to the size and extent of the primary tumor.
5. N Stage: Part of the TNM staging system, N stage refers to the involvement of nearby lymph nodes.
6. 6th Stage: Breast Adjusted AJCC 6th Stage.
7. Grade: Grading and differentiation codes of 1-4.
8. A Stage: SEER historic stage A.
9. Tumor Size: Information on tumor size. 
10. Estrogen Status: ER Status Recode Breast Cancer.
11. Progesterone Status: PR Status Recode Breast Cancer.
12. Regional Node Examined: Records the total number of regional lymph nodes that were removed and examined by the pathologist.
13. Regional Node Positive: Records the exact number of regional lymph nodes examined by the pathologist that were found to contain metastases. 
14. Survival Months - Number of survival months. Created using complete dates, including days, therefore may differ from survival time calculated from year and month only.
15. Status: Vital status recode (study cutoff used). Any patient that dies after the follow-up cut-off date is recoded to alive as of the cut-off date.
''')
container.markdown('#### :blue[The Dataset]')
container.dataframe(data)
container.download_button(
    label=':floppy_disk: Download Dataset',
    data=data.to_csv().encode('utf-8'),
    file_name=f'SEER_Breast_Cancer_Dataset.csv',
    mime='text/csv'
)
################################################################################################################################


#################################################################################################################################
# container 1 --- proportion of patient status, visualize in pie chart
container1 = st.container(border=True)
container1.markdown('### :blue[Proportion of Patient Status]')
custom_colors = ['#57a4fe', '#cff5ff']
status = data['Status'].value_counts()
pie = px.pie(status, values=status.values, names=['Alive', 'Dead'], color_discrete_sequence=custom_colors, height=600)
pie.update_traces(textfont=dict(size=25, color='black'))
pie.update_layout(template='plotly_dark')
container1.plotly_chart(pie, theme='streamlit', use_container_width=True)
##################################################################################################################################


##################################################################################################################################
# container 3 --- distribution of categorical variable, visualize in treemap
container3 = st.container(border=True)
container3.markdown('### :blue[Distribution of Categorical Variables]')
container3.markdown('### ')
col1, col2 = container3.columns(2)
var1 = col1.selectbox('Select first variable:', categorical_var)
var2 = col2.selectbox('Select second variable:', categorical_var)

if var1 == var2:
    fig = px.treemap(data, path=[var1], color_discrete_sequence=px.colors.qualitative.Plotly_r, title=f'Distribution of {var1}')
else:
    fig = px.treemap(data, path=[var1, var2], color_discrete_sequence=px.colors.qualitative.Plotly_r, title=f'Segmentation of {var1} by {var2}')

fig.update_layout(height=800)
fig.update_layout(title_font=dict(size=25))
fig.update_layout(template='plotly_dark')
container3.plotly_chart(fig, theme='streamlit', use_container_width=True)
####################################################################################################################################


#####################################################################################################################################
# container 4 --- distribution plot for numerical variables, visualize in box plot and histogram
container4 = st.container(border=True)
container4.markdown('### :blue[Distribution of Numerical Variables]')
container4.markdown('### ')

cat_option = [None, 'Race', 'Marital Status', 'T Stage', 'N Stage', '6th Stage', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status', 'Status']
num_option = ['Age', 'Tumor Size', 'Regional Node Examined', 'Regional Node Positive', 'Survival Months']

data_selected = container4.selectbox('Select the group for comparison', data_option.keys())
col1, col2 = container4.columns(2)
numvar_selected = col1.selectbox('Select a numerical variable', num_option)
catvar_selected = col2.selectbox('Select a categorical variable', cat_option)
var_list = [numvar_selected, catvar_selected]
plot = helper.plot_histogram(data_option[data_selected], var_list)
container4.plotly_chart(plot, theme='streamlit', use_container_width=True)
######################################################################################################################################


######################################################################################################################################
# container 5 --- correlation between numerical variables
container5 = st.container(border=True)
container5.markdown('### :blue[Correlation between Numerical Variables]')
container5.markdown('### ')
col3, col4 = container5.columns(2)
var1 = col3.selectbox('Select first variable', num_option)
var2 = col4.selectbox('Select second variable', num_option)
plot_scatter(data, var1, var2, container5)
#######################################################################################################################################


#######################################################################################################################################
# container 6 - statistical summary of the variables
container6 = st.container(border=True, height=1100)
container6.markdown('### :blue[Statistical Analysis of Variables]')
container6.markdown('### ')
container6.markdown('''
Continuous variables were represented by their median and IQR, while categorical values were presented as frequencies and proportions. 
Statistical significance was determined using the Student\'s t-test for continuous variables and the Chi-square test for categorical variables, 
with a p-value of less than 0.05 indicating significance.
''')
styled_summary_stat = summary_stat.style.set_table_styles([
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
#container6.write(styled_summary_stat.to_html(), unsafe_allow_html=True)
container6.table(styled_summary_stat)
#######################################################################################################################################