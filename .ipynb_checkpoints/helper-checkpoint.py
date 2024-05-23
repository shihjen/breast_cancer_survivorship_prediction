# function to visualize distribution of numerical variable
def plot_histogram(data, var_list):
    import plotly.express as px
    if var_list[1] == None:
        fig = px.histogram(data, x=var_list[0], marginal='box', height=800, histnorm='', 
                           color_discrete_sequence=px.colors.qualitative.Plotly, title=f'Distribution of {var_list[0]}')
    else:
        fig = px.histogram(data, x=var_list[0], color=var_list[1], marginal='box', height=800, histnorm='percent', 
                           color_discrete_sequence=px.colors.qualitative.Plotly, 
                           title=f'Distribution of {var_list[0]} by {var_list[1]}')
        
        
    fig.update_layout(template='plotly_dark', xaxis=dict(showticklabels=True))
    fig.update_layout(xaxis_title=var_list[0], yaxis_title='Percentage (%)', barmode='overlay', 
                      margin=dict(l=50, r=50, t=50, b=50),
                      legend=dict(orientation='h',  # Horizontal orientation
                                  yanchor='top',    # Aligns the top of the legend with the specified y position
                                  y=-0.2,           # Position the legend below the plot
                                  xanchor='center', # Center the legend horizontally
                                  x=0.5             # Center the legend horizontally
                                 ))
    return fig

# function to perform student t-test
def ttest(var, data):
    from scipy.stats import ttest_ind
    data_copy = data[~data[var].isna()]
    group_data = data_copy.groupby('Status')
    alive = group_data.get_group('Alive')
    dead = group_data.get_group('Dead')
    res = ttest_ind(alive[var], dead[var])
    test_stat = round(res[0], 4)
    pvalue = round(res[1], 4)
    return test_stat, pvalue

# function to perform chi2 test
def chi2test(var, data):
    import pandas as pd
    from scipy.stats import chi2_contingency
    contingency_table = pd.crosstab(data[var], data['Status'])
    res = chi2_contingency(contingency_table)
    test_stat = round(res[0], 4)
    pvalue = round(res[1], 4)
    return test_stat, pvalue

# helper function to perform hyperparameters tuning via grid search and return the tuned model
def hyperparameterTuning(model, param_grid, Xtrain, ytrain):
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    stratifiedCV = StratifiedKFold(n_splits=3)
    # create the GridSearchCV object
    grid_search = GridSearchCV(model,                  # model to be tuned
                               param_grid,             # search grid for the parameters
                               cv=stratifiedCV,        # stratified K-fold cross validation to evaluate the model performance
                               scoring='roc_auc',      # metric to assess the model performance, weighted F1 score (consider the proportion of classes in the dataset)
                               n_jobs=-1)              # use all cpu cores to speed-up CV search

    # fit the data into the grid search space
    grid_search.fit(Xtrain, ytrain)

    # print the best parameters and the corresponding ROC_AUC score
    print('Best Hyperparameters from Grid Search : ', grid_search.best_params_)
    print('Best AUROC Score: ', grid_search.best_score_)
    print()

    # get the best model
    best_model = grid_search.best_estimator_
    
    # return the hyperparameters tuned model
    return best_model

# function to get the performance of all trained models
def model_performance(models, X, y):
    from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
    import pandas as pd
    precision_list = []
    recall_list = []
    f1_list = []
    mcc_list = []
    cont = []
    for model_name, model in models.items():
        ypred = model.predict(X)
        precision = precision_score(y, ypred, average='macro')
        recall = recall_score(y, ypred, average='macro')
        f1 = f1_score(y, ypred, average='macro')
        mcc = matthews_corrcoef(y, ypred)
        res = [model_name, precision, recall, f1, mcc]
        cont.append(res)
    res_df = pd.DataFrame(cont, columns=['Model','Precision','Recall','F1','MCC'])
    return res_df

# function to plot confusion matrix
def plot_confusion_matrix(model, Xtrain, ytrain, Xtest, ytest):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    label = ['Alive', 'Dead']

    ypred_train = model.predict(Xtrain)
    ypred_test = model.predict(Xtest)

    cm_train = confusion_matrix(ytrain, ypred_train)
    cm_test = confusion_matrix(ytest, ypred_test)
    
    figure, axes = plt.subplots(1,2, figsize=(12,5))
    sns.heatmap(cm_train, annot=True, cmap='plasma', cbar=False, ax=axes[0], xticklabels=label, yticklabels=label, fmt='d')
    axes[0].set_title('Training Data')
    sns.heatmap(cm_test, annot=True, cmap='plasma', cbar=False, ax=axes[1], xticklabels=label, yticklabels=label, fmt='d')
    axes[1].set_title('Test Data')
    plt.tight_layout()
    plt.show()