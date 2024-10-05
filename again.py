import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import dash
from dash import Dash, dcc, callback, Output, Input,html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
df = pd.read_csv('Breast_Cancer.csv')   # df, data, salary ,.... suitable name
alive=df[df["Status"]=="Alive"].reset_index().drop(columns=["index"])
dead=df[df["Status"]=="Dead"].reset_index().drop(columns=["index"])
# frist figures for distripution 
#age
figh_1=px.histogram(df,x="Age",color='Status', barmode='group',
                   title="distrubution of Age by Status", histfunc='avg')
figh_1.update_layout(yaxis_title='Average Age')
# Race
figh_2=px.histogram(df,x="Race",color='Status', barmode='group',
                   title="distrubution of Race by Status", histfunc='avg')
figh_2.update_layout(yaxis_title='Average Race')
# Marital Status
figh_3=px.histogram(df,x="Marital Status",color='Status', barmode='group',
                   title="distrubution of Marital Status by Status", histfunc='avg')
figh_3.update_layout(yaxis_title='Average Marital Status')
# T Stage 
figh_4=px.histogram(df,x="T Stage ",color='Status', barmode='group',
                   title="distrubution of T Stage  by Status", histfunc='avg')
figh_4.update_layout(yaxis_title='Average T Stage')
# N Stage
figh_5=px.histogram(df,x="N Stage",color='Status', barmode='group',
                   title="distrubution of N Stage  by Status", histfunc='avg')
figh_5.update_layout(yaxis_title='Average N Stage')
# 6th Stage
figh_6=px.histogram(df,x="6th Stage",color='Status', barmode='group',
                   title="distrubution of 6th Stage  by Status", histfunc='avg')
figh_6.update_layout(yaxis_title='Average 6th Stage')
# differentiate
figh_7=px.histogram(df,x="differentiate",color='Status', barmode='group',
                   title="distrubution of differentiate  by Status", histfunc='avg')
figh_7.update_layout(yaxis_title='Average differentiate')
# A Stage
figh_8=px.histogram(df,x="A Stage",color='Status', barmode='group',
                   title="distrubution of A Stage  by Status", histfunc='avg')
figh_8.update_layout(yaxis_title='Average A Stage')
#Estrogen Status
figh_9=px.histogram(df,x="Estrogen Status",color='Status', barmode='group',
                   title="distrubution of Estrogen Status by Status", histfunc='avg')
figh_9.update_layout(yaxis_title='Estrogen Status')
#Progesterone Status
figh_10=px.histogram(df,x="Progesterone Status",color='Status', barmode='group',
                   title="distrubution of Progesterone Status  by Status", histfunc='avg')
figh_10.update_layout(yaxis_title='Average Progesterone Status')
#Grade
figh_11=px.histogram(df,x="Grade",color='Status', barmode='group',
                   title="distrubution of Grade  by Status", histfunc='avg')
figh_11.update_layout(yaxis_title='Average Grade')
# function for choocing the figure in dash 

    
# comparing  with pie chart ... alive or dead 
#race
fig_r1 = px.pie(df[df["Status"]=="Alive"], names='Race')
fig_r2 = px.pie(df[df["Status"]=="Dead"], names='Race')
Race_fig = make_subplots(rows=1, cols=2, subplot_titles=("Race with distribution of alive statue", "Race with distribution of dead statue"),specs=[[{'type': 'pie'}, {'type': 'pie'}]])
Race_fig.add_trace(fig_r1.data[0], row=1, col=1)
Race_fig.add_trace(fig_r2.data[0], row=1, col=2)
# Marital Status
fig_M1 = px.pie(df[df["Status"]=="Alive"], names='Marital Status')
fig_M2 = px.pie(df[df["Status"]=="Dead"], names='Marital Status')
Marital_fig = make_subplots(rows=1, cols=2, subplot_titles=("Marital Status with distribution of alive statue", "Marital Status with distribution of dead statue"),specs=[[{'type': 'pie'}, {'type': 'pie'}]])
Marital_fig.add_trace(fig_M1.data[0], row=1, col=1)
Marital_fig.add_trace(fig_M2.data[0], row=1, col=2)
# T Stage
fig_t1 = px.pie(df[df["Status"]=="Alive"], names='T Stage ')
fig_t2 = px.pie(df[df["Status"]=="Dead"], names='T Stage ')
T_Stage_fig = make_subplots(rows=1, cols=2, subplot_titles=("T Stage with distribution of alive statue", "T Stage with distribution of dead statue"),specs=[[{'type': 'pie'}, {'type': 'pie'}]])
T_Stage_fig.add_trace(fig_t1.data[0], row=1, col=1)
T_Stage_fig.add_trace(fig_t2.data[0], row=1, col=2)
# N Stage
fig_n1 = px.pie(df[df["Status"]=="Alive"], names='N Stage')
fig_n2 = px.pie(df[df["Status"]=="Dead"], names='N Stage')
N_Stage_fig = make_subplots(rows=1, cols=2, subplot_titles=("N Stage with distribution of alive statue", "N Stage with distribution of dead statue"),specs=[[{'type': 'pie'}, {'type': 'pie'}]])
N_Stage_fig.add_trace(fig_n1.data[0], row=1, col=1)
N_Stage_fig.add_trace(fig_n2.data[0], row=1, col=2)
# 6th Stage 
fig_6th1 = px.pie(df[df["Status"]=="Alive"], names='6th Stage')
fig_6th2 = px.pie(df[df["Status"]=="Dead"], names='6th Stage')
th6_fig = make_subplots(rows=1, cols=2, subplot_titles=("6th Stage with distribution of alive statue", "6th Stage with distribution of dead statue"),specs=[[{'type': 'pie'}, {'type': 'pie'}]])
th6_fig.add_trace(fig_6th1.data[0], row=1, col=1)
th6_fig.add_trace(fig_6th2.data[0], row=1, col=2)
# differentiate
fig_differentiate1 = px.pie(df[df["Status"]=="Alive"], names='differentiate')
fig_differentiate2 = px.pie(df[df["Status"]=="Dead"], names='differentiate')
differentiate_fig = make_subplots(rows=1, cols=2, subplot_titles=("differentiate with distribution of alive statue", "differentiate with distribution of dead statue"),specs=[[{'type': 'pie'}, {'type': 'pie'}]])
differentiate_fig.add_trace(fig_differentiate1.data[0], row=1, col=1)
differentiate_fig.add_trace(fig_differentiate2.data[0], row=1, col=2)
# A Stage
fig_AStage1 = px.pie(df[df["Status"]=="Alive"], names='A Stage')
fig_AStage2 = px.pie(df[df["Status"]=="Dead"], names='A Stage')
AStage_fig = make_subplots(rows=1, cols=2, subplot_titles=("A Stage with distribution of alive statue", "A Stage with distribution of dead statue"),specs=[[{'type': 'pie'}, {'type': 'pie'}]])
AStage_fig.add_trace(fig_AStage1.data[0], row=1, col=1)
AStage_fig.add_trace(fig_AStage2.data[0], row=1, col=2)
# Estrogen Status
fig_Estrogen1 = px.pie(df[df["Status"]=="Alive"], names='A Stage')
fig_Estrogen2 = px.pie(df[df["Status"]=="Dead"], names='A Stage')
Estrogen_fig = make_subplots(rows=1, cols=2, subplot_titles=("Estrogen Status with distribution of alive statue", "Estrogen Status with distribution of dead statue"),specs=[[{'type': 'pie'}, {'type': 'pie'}]])
Estrogen_fig.add_trace(fig_Estrogen1.data[0], row=1, col=1)
Estrogen_fig.add_trace(fig_Estrogen2.data[0], row=1, col=2)
# Progesterone Status
fig_Progesterone1 = px.pie(df[df["Status"]=="Alive"], names='A Stage')
fig_Progesterone2 = px.pie(df[df["Status"]=="Dead"], names='A Stage')
Progesterone_fig = make_subplots(rows=1, cols=2, subplot_titles=("Progesterone Status with distribution of alive statue", "Progesterone Status with distribution of dead statue"),specs=[[{'type': 'pie'}, {'type': 'pie'}]])
Progesterone_fig.add_trace(fig_Progesterone1.data[0], row=1, col=1)
Progesterone_fig.add_trace(fig_Progesterone2.data[0], row=1, col=2)

#____#


#___# BOX PLOT 
#Tumor Size
fig_Tm = px.box(df, x='Status', y='Tumor Size', 
             color='Status',
             color_discrete_sequence=['blue', 'red'],  # Custom colors for the categories
             category_orders={'Status': ["Alive","Dead"]})
# Regional Node Examined
fig_RNE = px.box(df, x='Status', y='Regional Node Examined', 
             color='Status',
             color_discrete_sequence=['blue', 'red'],  # Custom colors for the categories
             category_orders={'Status': ["Alive","Dead"]})
# Reginol Node Positive
fig_RNP = px.box(df, x='Status', y='Reginol Node Positive', 
             color='Status',
             color_discrete_sequence=['blue', 'red'],  # Custom colors for the categories
             category_orders={'Status': ["Alive","Dead"]})
# Survival Months
fig_M = px.box(df, x='Status', y='Survival Months', 
             color='Status',
             color_discrete_sequence=['blue', 'red'],  # Custom colors for the categories
             category_orders={'Status': ["Alive","Dead"]})

#___# SCATTER
# Age Vs Tumor Size
fig_at = px.scatter(
    df,x='Age',y='Tumor Size',
    color='Status',  
    color_discrete_map={'Alive': 'blue', 'Dead': 'red'},  # Custom colors
    title="Scatter Plot of Age vs. Tumor Size by Status")
# Age Vs Regional Node Examined
fig_are = px.scatter(
    df,x='Age',y='Regional Node Examined',
    color='Status',  
    color_discrete_map={'Alive': 'blue', 'Dead': 'red'},  # Custom colors
    title="Scatter Plot of Age vs. Regional Node Examined by Status")
# Age Vs Reginol Node Positive
fig_arp = px.scatter(
    df,x='Age',y='Reginol Node Positive',
    color='Status',  
    color_discrete_map={'Alive': 'blue', 'Dead': 'red'},  # Custom colors
    title="Scatter Plot of Age vs. Reginol Node Positive by Status")
# Age Vs Survival Months
fig_sm = px.scatter(
    df,x='Age',y='Survival Months',
    color='Status',  
    color_discrete_map={'Alive': 'blue', 'Dead': 'red'},  # Custom colors
    title="Scatter Plot of Age vs. Survival Months by Status")
# Regional Node Examined Vs Reginol Node Positive
fig_nep = px.scatter(
    df,x='Regional Node Examined',y='Reginol Node Positive',
    color='Status',  
    color_discrete_map={'Alive': 'blue', 'Dead': 'red'},  # Custom colors
    title="Scatter Plot of Regional Node Examined vs. Reginol Node Positive by Status")
# Regional Node Positive Vs Survival Months
fig_nps = px.scatter(
    df,x='Reginol Node Positive',y='Survival Months',
    color='Status',  
    color_discrete_map={'Alive': 'blue', 'Dead': 'red'},  # Custom colors
    title="Scatter Plot of Reginol Node Positive vs. Survival Months by Status")
# Regional Node Examined Vs Survival Months
fig_nes = px.scatter(
    df,x='Regional Node Examined',y='Survival Months',
    color='Status',  
    color_discrete_map={'Alive': 'blue', 'Dead': 'red'},  # Custom colors
    title="Scatter Plot of Regional Node Examined vs. Survival Months by Status")
 
 
 
# FUNCTION 

    
## START DASH 
def hist_figures(hist_name):
    if hist_name == "distrubution of Age by Status" : 
        return figh_1
    elif hist_name == "distrubution of Race by Status" :
        return figh_2
    elif hist_name == "distrubution of Marital Status by Status" : 
        return figh_3
    elif hist_name == "distrubution of T Stage by Status" : 
        return figh_4 
    elif hist_name == "distrubution of N Stage by Status" : 
        return figh_5
    elif hist_name == "distrubution of 6th Stage by Status" : 
        return figh_6
    elif hist_name == "distrubution of differentiate by Status" : 
        return figh_7
    elif hist_name == "distrubution of A Stage by Status" : 
        return figh_8
    elif hist_name == "distrubution of Estrogen Status by Status" : 
        return figh_9
    elif hist_name == "distrubution of Progesterone Status by Status" : 
        return figh_10
    elif hist_name == "distrubution of Grade by Status" : 
        return figh_11

def pie_figures(pie_name):
    if pie_name == "distrubution for Race by statue" : 
        return Race_fig
    elif pie_name == "distrubution for Marital Status by statue" :
        return Marital_fig
    elif pie_name == "distrubution for T Stag by statue" : 
        return T_Stage_fig
    elif pie_name == "distrubution of N Stage by statue" : 
        return N_Stage_fig 
    elif pie_name == "distrubution of 6th Stage by statue" : 
        return th6_fig
    elif pie_name == "distrubution of differentiate by statue" : 
        return differentiate_fig
    elif pie_name == "distrubution of A Stage by statue" : 
        return AStage_fig
    elif pie_name == "distrubution of Estrogen Status by statue" : 
        return Estrogen_fig
    elif pie_name == "distrubution of Progesterone Status use by statue" : 
        return Progesterone_fig
 
def box_figures(box_name):
    if box_name == "distrubution for Tumor Size by statue" : 
        return fig_Tm
    elif box_name == "distrubution for Regional Node Examined by statue" :
        return fig_RNE
    elif box_name == "distrubution for Reginol Node Positive by statue" : 
        return fig_RNP
    elif box_name == "distrubution ofSurvival Months by statue" : 
        return fig_M 

def scatter_figure(scatter_name):
    if scatter_name == "Age vs. Tumor Size" : 
        return fig_at
    elif scatter_name == "Age vs. Regional Node Examined" : 
        return fig_are 
    elif scatter_name == "Age vs. Reginol Node Positive by Status" : 
        return fig_arp
    elif scatter_name == "Age vs. Survival Months by Status" : 
        return fig_sm
    elif scatter_name == "Regional Node Examined vs. Reginol Node Positive" : 
        return fig_nep
    elif scatter_name == "Reginol Node Positive vs. Survival Months" : 
        return fig_nps
    elif scatter_name == "Regional Node Examined vs. Survival Months" : 
        return fig_nes
    

    
    
## START DASH 
app=Dash()
app.layout=html.Div([
    html.Div([
        html.H1("Comparative Histogram Analysis")]),
    html.Div([
        dcc.RadioItems(
            id='frist-figure-selector',
            options=[{'label': key, 'value': key} for key in 
                     ["distrubution of Age by Status" ,"distrubution of Race by Status",
                      "distrubution of Marital Status by Status","distrubution of T Stage by Status",
                      "distrubution of N Stage by Status","distrubution of 6th Stage by Status",
                      "distrubution of differentiate by Status","distrubution of A Stage by Status",
                      "distrubution of Estrogen Status by Status","distrubution of Progesterone Status by Status",
                      "distrubution of Grade by Status"]],  # First three figures
            value='distrubution of Age by Status'
        ),
        dcc.Graph(id='frist-figure'),
    ]),
    html.Div([
            html.H1("Comparative pie chart Analysis")])
    ,
    html.Div([
        dcc.RadioItems(
            id='second-figure-selector',
            options=[{'label': key, 'value': key} for key in 
                     ["distrubution for Race by statue","distrubution for Marital Status by statue",
                      "distrubution for T Stag by statue","distrubution of N Stage by statue",
                      "distrubution of 6th Stage by statue","distrubution of differentiate by statue",
                      "distrubution of A Stage by statue","distrubution of Estrogen Status by statue",
                      "distrubution of Progesterone Status use by statue"]],  # First three figures
            value='distrubution for Race by statue'
        ),
        dcc.Graph(id='second-figure'),
    ]),
    html.Div([
            html.H1("Comparative box plot Analysis")])
    ,
    html.Div([
        dcc.RadioItems(
            id='third-figure-selector',
            options=[{'label': key, 'value': key} for key in 
                     ["distrubution for Tumor Size by statue","distrubution for Regional Node Examined by statue",
                      "distrubution for Reginol Node Positive by statue","distrubution ofSurvival Months by statue"]],  # First three figures
            value="distrubution for Tumor Size by statue"
        ),
        dcc.Graph(id='third-figure'),
    ]),
    html.Div([
            html.H1("Comparative scatter plot Analysis")]),
    html.Div([
        dcc.RadioItems(
            id='final-figure-selector',
            options=[{'label': key, 'value': key} for key in 
                     ["Age vs. Tumor Size","Age vs. Regional Node Examined",
                       "Age vs. Reginol Node Positive by Status","Age vs. Survival Months by Status",
                       "Regional Node Examined vs. Reginol Node Positive","Reginol Node Positive vs. Survival Months",
                       "Regional Node Examined vs. Survival Months"]],  # First three figures
            value="Age vs. Tumor Size"
        ),
        dcc.Graph(id='final-figure'),
    ])    
    
])
# Callback to update the frist figure based on selection
@app.callback(
    Output('frist-figure', 'figure'),
    Input('frist-figure-selector', 'value')
)
def update_frist_figure(selected_figure):
    return hist_figures(selected_figure)

@app.callback(
    Output('second-figure', 'figure'),
    Input('second-figure-selector', 'value')
)
def update_second_figure(selected_figure):
    return pie_figures(selected_figure)

# Callback to update the third figure based on selection
@app.callback(
    Output('third-figure', 'figure'),
    Input('third-figure-selector', 'value')
)
def update_third_figure(selected_figure):
    return box_figures(selected_figure)

@app.callback(
    Output('final-figure', 'figure'),
    Input('final-figure-selector', 'value')
)
def update_final_figure(selected_figure):
    return scatter_figure(selected_figure)

if __name__ == '__main__':
    app.run_server(debug=True)




