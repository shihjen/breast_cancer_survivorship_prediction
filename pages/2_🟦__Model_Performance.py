# import the required dependencies
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

# Streamlit page configuration
st.set_page_config(
    page_title='Model_Performance',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title(':blue[Model Performance]')


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

# load the dataset
training_data = pd.read_csv('data/training_data.csv')
test_data = pd.read_csv('data/test_data.csv')

Xtrain = training_data.drop(columns=['Status'], axis=1)
ytrain = training_data['Status']
Xtest = test_data.drop(columns=['Status'], axis=1)
ytest = test_data['Status']

##############################################################################################################
# container_pre - learning algorithms used
container_pre = st.container(border=True)
container_pre.markdown('### :blue[Learning Algorithms]')
container_pre.image('image/learning_algorithms_black.png', use_column_width=True)
##############################################################################################################


##############################################################################################################
# container - key metrics: precision, recall, F1 and MCC
container = st.container(border=True)
modelPerformance = pd.read_csv('data/model_performance_base_result.csv')
container.markdown('### :blue[Precision, Recall, F1 Score and MCC of Trained Models]')
container.markdown('### ')
fig = make_subplots(rows=4, cols=1, shared_yaxes=True)
fig.add_trace(go.Bar(x=modelPerformance.Model, y=modelPerformance.Precision, name='Precision', marker=dict(color='#57a4fe'), text=round(modelPerformance.Precision,4), textposition='outside'), row=1, col=1)
fig.add_trace(go.Bar(x=modelPerformance.Model, y=modelPerformance.Recall, name='Recall', marker=dict(color='#cff5ff'), text=round(modelPerformance.Recall,4), textposition='outside'), row=2, col=1)
fig.add_trace(go.Bar(x=modelPerformance.Model, y=modelPerformance.F1, name='F1', marker=dict(color='#8ac9fe'), text=round(modelPerformance.F1,4), textposition='outside'), row=3, col=1)
fig.add_trace(go.Bar(x=modelPerformance.Model, y=modelPerformance.MCC, name='MCC', marker=dict(color='#340a5f'), text=round(modelPerformance.MCC,4), textposition='outside'), row=4, col=1)
fig.update_layout(height=1600)
# add titles for each subplot
fig.update_yaxes(title_text='Precision', row=1, col=1)
fig.update_yaxes(title_text='Recall', row=2, col=1)
fig.update_yaxes(title_text='F1', row=3, col=1)
fig.update_yaxes(title_text='MCC', row=4, col=1)
fig.update_xaxes(title_text='Model', row=4, col=1)
fig.update_layout(template='plotly_dark')
container.plotly_chart(fig, theme='streamlit', use_container_width=True)
#################################################################################################################


#################################################################################################################
# container 2 - ROC Curve
container2 = st.container(border=True)
container2.markdown('### :blue[Receiver Operating Characteristic (ROC) Curve]')
container2.markdown('### ')
ROC_curve = pd.read_csv('data/ROC_curve.csv')
auc_df = pd.read_csv('data/auc_result.csv')

# Create a trace for the ROC curve
trace = go.Scatter(
    x=ROC_curve.lr_fpr, 
    y=ROC_curve.lr_tpr, 
    mode='lines', 
    name=f'Logistic Regression (AUC = {auc_df["Logistic Regression"].values[0]:.2f})',
    line=dict(color='#57a4fe')
)

trace2 = go.Scatter(
    x=ROC_curve.tree_fpr, 
    y=ROC_curve.tree_tpr, 
    mode='lines', 
    name=f'Decison Tree (AUC = {auc_df["Decision Tree"].values[0]:.2f})',
    line=dict(color='green')
)

trace3 = go.Scatter(
    x=ROC_curve.rf_fpr, 
    y=ROC_curve.rf_tpr, 
    mode='lines', 
    name=f'Random Forest (AUC = {auc_df["Random Forest"].values[0]:.2f})',
    line=dict(color='#c63d4d')
)

trace4 = go.Scatter(
    x=ROC_curve.svc_fpr, 
    y=ROC_curve.svc_tpr, 
    mode='lines', 
    name=f'Support Vector Machine (AUC = {auc_df["Support Vector Machine"].values[0]:.2f})',
    line=dict(color='red')
)

trace5 = go.Scatter(
    x=ROC_curve.gbc_fpr, 
    y=ROC_curve.gbc_tpr, 
    mode='lines', 
    name=f'Gradient Boosting (AUC = {auc_df["Gradient Boosting"].values[0]:.2f})',
    line=dict(color='#f2ea69')
)

# create a trace for the diagonal line
diagonal_trace = go.Scatter(
    x=[0, 1], 
    y=[0, 1], 
    mode='lines', 
    name='Random guess',
    line=dict(color='white', dash='dash')
)

# create the figure
fig2 = go.Figure()

# add traces to the figure
fig2.add_trace(trace)
fig2.add_trace(trace2)
fig2.add_trace(trace3)
fig2.add_trace(trace4)
fig2.add_trace(trace5)
fig2.add_trace(diagonal_trace)

# update the layout
fig2.update_layout(
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    xaxis=dict(constrain='domain'),
    yaxis=dict(constrain='domain'),
    showlegend=True,
    height=800
)

fig2.update_layout(template='plotly_dark')
container2.plotly_chart(fig2, theme='streamlit', use_container_width=True)
########################################################################################################################



########################################################################################################################
# container 3 - precision recall curve
container3 = st.container(border=True)
container3.markdown('### :blue[Precision Recall Curve]')
container3.markdown('### ')
ap_df = pd.read_csv('data/average_precision_result.csv')
pr_res_df = pd.read_csv('data/presion_recall_curve.csv')

trace = go.Scatter(
    x=pr_res_df.lr_recall, 
    y=pr_res_df.lr_precision, 
    mode='lines', 
    name=f'Logistic Regression (AP = {ap_df["Logistic Regression"].values[0]:.2f})',
    line=dict(color='#57a4fe')
)

trace2 = go.Scatter(
    x=pr_res_df.tree_recall, 
    y=pr_res_df.tree_precision, 
    mode='lines', 
    name=f'Decision Tree (AP = {ap_df["Decision Tree"].values[0]:.2f})',
    line=dict(color='green')
)

trace3 = go.Scatter(
    x=pr_res_df.rf_recall, 
    y=pr_res_df.rf_precision, 
    mode='lines', 
    name=f'Random Forest (AP = {ap_df["Random Forest"].values[0]:.2f})',
    line=dict(color='#c63d4d')
)

trace4 = go.Scatter(
    x=pr_res_df.svc_recall, 
    y=pr_res_df.svc_precision, 
    mode='lines', 
    name=f'Support Vector Machine (AP = {ap_df["Support Vector Machine"].values[0]:.2f})',
    line=dict(color='red')
)

trace5 = go.Scatter(
    x=pr_res_df.gbc_recall, 
    y=pr_res_df.gbc_precision, 
    mode='lines', 
    name=f'Gradient Boosting (AP = {ap_df["Gradient Boosting"].values[0]:.2f})',
    line=dict(color='#f2ea69')
)

fig3 = go.Figure()
fig3.add_trace(trace)
fig3.add_trace(trace2)
fig3.add_trace(trace3)
fig3.add_trace(trace4)
fig3.add_trace(trace5)

# Update the layout
fig3.update_layout(
    xaxis_title='Recall',
    yaxis_title='Precision',
    xaxis=dict(constrain='domain'),
    yaxis=dict(constrain='domain'),
    showlegend=True,
    height=800
)

fig3.update_layout(template='plotly_dark')
container3.plotly_chart(fig3, theme='streamlit', use_container_width=True)
#########################################################################################################################



#########################################################################################################################
# container 4 - confusion matrix
container4 = st.container(border=True)
container4.markdown('### :blue[Confusion Matrix]')
container4.markdown('### ')
model_choice = container4.selectbox('Select a trained model:', model_option.keys())

def plot_confusion_matrix(model):
    label = ['Alive', 'Dead']
    ypred_train = model.predict(Xtrain)
    ypred_test = model.predict(Xtest)
    cm_train = confusion_matrix(ytrain, ypred_train, normalize='true')
    cm_test = confusion_matrix(ytest, ypred_test, normalize='true')
    figure, axes = plt.subplots(1,2, figsize=(12,5))
    figure.patch.set_facecolor('black')  # Set the figure background color
    sns.set_style('dark')
    sns.heatmap(cm_train, annot=True, cmap='Blues', cbar=False, ax=axes[0], xticklabels=label, yticklabels=label)
    axes[0].set_title('Training Data', color='white')
    sns.heatmap(cm_test, annot=True, cmap='Blues', cbar=False, ax=axes[1], xticklabels=label, yticklabels=label)
    axes[1].set_title('Test Data', color='white')
    # Set the tick labels color to white
    for tick_label in axes[0].get_xticklabels():
        tick_label.set_color('white')
    for tick_label in axes[0].get_yticklabels():
        tick_label.set_color('white')
    for tick_label in axes[1].get_xticklabels():
        tick_label.set_color('white')
    for tick_label in axes[1].get_yticklabels():
        tick_label.set_color('white')
        plt.show()
    container4.pyplot(plt)

plot_confusion_matrix(model_option[model_choice])
########################################################################################################################
