import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

# Load your dataset
df = pd.read_csv("Breast_Cancer.csv")  # Replace with your actual data path

# Set up Streamlit page configuration
st.set_page_config(page_title="Breast Cancer Data EDA", layout="wide")

# Sidebar for navigation
st.sidebar.title("Breast Cancer Data EDA")
option = st.sidebar.selectbox("Choose an EDA technique:", 
                                ["Distribution of Numerical Features", 
                                 "Proportions of Categorical Features", 
                                 "Summary of Numerical Features", 
                                 "Relationships Between Features", 
                                 "Key Findings Summary"])

# Add an empty Machine Learning section
st.sidebar.subheader("Machine Learning")
st.sidebar.write("This section will contain Machine Learning analysis.")

# Display title
st.title("Exploratory Data Analysis (EDA) of Breast Cancer Dataset")

# Distribution of Numerical Features (Histograms)
if option == "Distribution of Numerical Features":
    st.subheader("Distribution of Numerical Features by Status")
    col = st.selectbox('Select a numerical feature:', df.select_dtypes(include='number').columns)
    if col:
        fig = px.histogram(df, x=col, color='Status', barmode='group', histfunc='avg', 
                           title=f"Distribution of {col} by Status")
        fig.update_layout(yaxis_title=f'Average {col}')
        st.plotly_chart(fig)

# Proportions of Categorical Features (Pie Charts)
elif option == "Proportions of Categorical Features":
    st.subheader("Proportions of Categorical Features by Status")
    feature = st.selectbox('Select a categorical feature:', 
                            ["Race", "Marital Status", "T Stage", "N Stage", "6th Stage", 
                             "differentiate", "A Stage", "Estrogen Status", "Progesterone Status"])
    if feature:
        fig_alive = px.pie(df[df["Status"] == "Alive"], names=feature, title=f"{feature}: Alive")
        fig_dead = px.pie(df[df["Status"] == "Dead"], names=feature, title=f"{feature}: Dead")
        pie_fig = make_subplots(rows=1, cols=2, subplot_titles=(f"{feature}: Alive", f"{feature}: Dead"), 
                                specs=[[{'type': 'pie'}, {'type': 'pie'}]])
        pie_fig.add_trace(fig_alive.data[0], row=1, col=1)
        pie_fig.add_trace(fig_dead.data[0], row=1, col=2)
        st.plotly_chart(pie_fig)

# Summary of Numerical Features (Box Plots)
elif option == "Summary of Numerical Features":
    st.subheader("Summary of Numerical Features by Status")
    feature = st.selectbox('Select a numerical feature for summary:', 
                            ["Tumor Size", "Regional Node Examined", "Reginol Node Positive", "Survival Months"])
    if feature:
        fig = px.box(df, x='Status', y=feature, color='Status', title=f"Summary of {feature} by Status", 
                     color_discrete_sequence=['blue', 'red'], 
                     category_orders={'Status': ["Alive", "Dead"]})
        st.plotly_chart(fig)

# Relationships Between Features (Scatter Plots)
elif option == "Relationships Between Features":
    st.subheader("Relationships Between Numerical Features")
    # Filter to get only numerical columns
    numerical_features = df.select_dtypes(include='number').columns.tolist()
    x_feature = st.selectbox('Select X-axis feature:', numerical_features)
    y_feature = st.selectbox('Select Y-axis feature:', numerical_features)
    if x_feature and y_feature:
        fig = px.scatter(df, x=x_feature, y=y_feature, color='Status', 
                         title=f"Relationship Between {x_feature} and {y_feature} by Status", 
                         color_discrete_map={'Alive': 'blue', 'Dead': 'red'})
        st.plotly_chart(fig)

# Key Findings Summary
elif option == "Key Findings Summary":
    st.subheader("Key Findings Summary")
    st.write("Key Findings:")
    st.write(f"1. Average age of patients: {df['Age'].mean():.2f} years")
    st.write(f"2. Overall survival rate: {(df['Status'] == 'Alive').mean() * 100:.2f}%")
    st.write(f"3. Median survival time: {df['Survival Months'].median()} months")
    st.write(f"4. Most common cancer stage: {df['6th Stage'].mode()[0]}")
    st.write(f"5. Correlation between tumor size and survival months: {df['Tumor Size'].corr(df['Survival Months']):.2f}")

    # Visualizations for Key Findings
    # Visualization 1: Survival Months by Cancer Stage
    st.subheader("Survival Months by Cancer Stage and Status")
    fig = px.box(df, x="6th Stage", y="Survival Months", color="Status",
                 title="Survival Months by Cancer Stage and Status")
    st.plotly_chart(fig)
    st.write("""
    **Conclusion:** 
    - There is a clear correlation between cancer stage and survival rates.
    - Stage IIA has the highest survival rate (approximately 80%).
    - Survival rates progressively decrease with higher stages, with IIIC having the lowest survival rate (around 60%).
    - For all stages, patients who survive typically have higher survival months compared to those who don't.
    """)

    # Visualization 2: Survival Rate by Cancer Stage
    st.subheader("Survival Rate by Cancer Stage")
    stage_survival = df.groupby('6th Stage').agg({
        'Status': lambda x: (x == 'Alive').mean() * 100
    }).reset_index()
    fig = px.bar(stage_survival, x='6th Stage', y='Status',
                 title="Survival Rate by Cancer Stage",
                 labels={'Status': 'Survival Rate (%)'})
    st.plotly_chart(fig)

    # Visualization 3: Correlation Heatmap
    st.subheader("Correlation Heatmap of Numerical Variables")
    correlation_matrix = df[['Age', 'Tumor Size', 'Regional Node Examined', 
                             'Reginol Node Positive', 'Survival Months']].corr()
    fig = px.imshow(correlation_matrix,
                    title="Correlation Heatmap of Numerical Variables")
    st.plotly_chart(fig)

    # Visualization 4: Age vs Tumor Size
    st.subheader("Age vs Tumor Size by Stage and Nodal Involvement")
    fig = px.scatter(df, x="Age", y="Tumor Size", 
                     color="6th Stage", size="Reginol Node Positive",
                     hover_data=['Status'],
                     title="Age vs Tumor Size by Stage and Nodal Involvement")
    st.plotly_chart(fig)
    st.write("""
    **Conclusion:** 
    - There appears to be no strong correlation between age and tumor size.
    - The age distribution is fairly similar across different cancer stages.
    - Most patients are between 40-70 years old, with the median age around 55-60.
    - There's a slight trend of IIIB stage being diagnosed in slightly older patients.
    """)

    # Visualization 5: Hormone Status Survival Months
    st.subheader("Survival Months by Hormone Receptor Status")
    df['Hormone_Status'] = df['Estrogen Status'] + '_' + df['Progesterone Status']
    fig = px.box(df, x="Hormone_Status", y="Survival Months", color="Status",
                 title="Survival Months by Hormone Receptor Status")
    st.plotly_chart(fig)
    st.write("""
    **Conclusion:** 
    - Patients with Positive/Positive hormone status (both Estrogen and Progesterone positive) tend to have better survival outcomes.
    - Negative/Negative hormone status appears to have the worst outcomes.
    - The hormone status, grade, and survival are interconnected, with certain combinations having better outcomes.
    """)

    # Visualization 6: Tumor Size vs Positive Nodes by T Stage
    st.subheader("Tumor Size vs Positive Nodes by T Stage")
    fig = px.scatter(df, x="Tumor Size", y="Reginol Node Positive",
                    color="T Stage ",  # Note the space after "Stage"
                    title="Tumor Size vs Positive Nodes by T Stage")
    st.plotly_chart(fig)
    st.write("""
    **Conclusion:** 
    - There appears to be a relationship between tumor size and the number of positive lymph nodes.
    - As T stage increases (T1 to T4), the distribution of positive nodes shifts upward.
    - T1 tumors typically have fewer positive nodes, while T3 and T4 tumors are associated with more nodal involvement.
    - The relationship shows considerable variation across different T stages.
    """)

    # Visualization 7: Cancer Stage Distribution by Race
    st.subheader("Cancer Stage Distribution by Race")
    stage_race = pd.crosstab(df['Race'], df['6th Stage'], normalize='index') * 100
    stage_race_long = stage_race.reset_index().melt(id_vars=['Race'], 
                                                    var_name='Stage', 
                                                    value_name='Percentage')
    fig = px.bar(stage_race_long, x="Race", y="Percentage", color="Stage",
                 title="Cancer Stage Distribution by Race")
    st.plotly_chart(fig)
    st.write("""
    **Conclusion:** 
    - There are some differences in cancer stage distribution among racial groups.
    - Black patients appear to have a slightly higher proportion of more advanced stages (IIIB and IIIC) compared to White and Other racial categories.
    - The overall distribution pattern is similar across races, but with subtle variations that might be clinically significant.
    """)

# Streamlit app runs here
