import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set up Streamlit page configuration
st.set_page_config(page_title="Breast Cancer Analysis", layout="wide")

# Load your dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Breast_Cancer.csv")
    return df

df = load_data()

# Sidebar for navigation
st.sidebar.title("Breast Cancer Analysis")
option = st.sidebar.selectbox(
    "Choose Analysis Type:",
    [
        "Distribution of Numerical Features",
        "Proportions of Categorical Features",
        "Summary of Numerical Features",
        "Relationships Between Features",
        "Key Findings Summary",
        "Machine Learning"
    ],
)

# Display title
st.title("Breast Cancer Dataset Analysis")

# Machine Learning functions
@st.cache_data
def prepare_data(df):
    # Encode target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Status'])
    
    # Prepare features
    X = df.drop('Status', axis=1)
    
    # Split features into numerical and categorical
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(drop='first', sparse_output=False))
            ]), categorical_features)
        ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, preprocessor, label_encoder, numerical_features, categorical_features

@st.cache_resource
def train_models(_X_train, _y_train, _preprocessor):
    models = {
        'Logistic Regression': Pipeline([
            ('preprocessor', _preprocessor),
            ('classifier', LogisticRegression(max_iter=1000))
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', _preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }
    
    for name, model in models.items():
        model.fit(_X_train, _y_train)
    
    return models

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }

# Distribution of Numerical Features
if option == "Distribution of Numerical Features":
    st.subheader("Distribution of Numerical Features by Status")
    col = st.selectbox(
        "Select a numerical feature:", 
        df.select_dtypes(include=['int64', 'float64']).columns
    )
    if col:
        fig = px.histogram(
            df,
            x=col,
            color="Status",
            barmode="group",
            histfunc="count",
            title=f"Distribution of {col} by Status",
        )
        st.plotly_chart(fig)

# Proportions of Categorical Features
elif option == "Proportions of Categorical Features":
    st.subheader("Proportions of Categorical Features by Status")
    feature = st.selectbox(
        "Select a categorical feature:",
        [col for col in df.columns if df[col].dtype == 'object' and col != 'Status']
    )
    if feature:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(f"{feature}: Alive", f"{feature}: Dead"),
            specs=[[{"type": "pie"}, {"type": "pie"}]]
        )
        
        for i, status in enumerate(['Alive', 'Dead']):
            status_data = df[df['Status'] == status][feature].value_counts()
            fig.add_trace(
                px.pie(values=status_data.values, names=status_data.index).data[0],
                row=1, col=i+1
            )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig)

# Summary of Numerical Features
elif option == "Summary of Numerical Features":
    st.subheader("Summary of Numerical Features by Status")
    feature = st.selectbox(
        "Select a numerical feature for summary:",
        df.select_dtypes(include=['int64', 'float64']).columns
    )
    if feature:
        fig = px.box(
            df,
            x="Status",
            y=feature,
            color="Status",
            title=f"Summary of {feature} by Status",
            color_discrete_sequence=["blue", "red"],
        )
        st.plotly_chart(fig)

# Relationships Between Features
elif option == "Relationships Between Features":
    st.subheader("Relationships Between Numerical Features")
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    x_feature = st.selectbox("Select X-axis feature:", numerical_features)
    y_feature = st.selectbox("Select Y-axis feature:", numerical_features)
    if x_feature and y_feature:
        fig = px.scatter(
            df,
            x=x_feature,
            y=y_feature,
            color="Status",
            title=f"Relationship Between {x_feature} and {y_feature} by Status",
        )
        st.plotly_chart(fig)

# Key Findings Summary
elif option == "Key Findings Summary":
    st.subheader("Key Findings Summary")
    
    # Display key statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Age", f"{df['Age'].mean():.1f} years")
    with col2:
        survival_rate = (df['Status'] == 'Alive').mean() * 100
        st.metric("Overall Survival Rate", f"{survival_rate:.1f}%")
    with col3:
        st.metric("Median Survival Time", f"{df['Survival Months'].median()} months")
    
    # Most common cancer stage
    st.write(f"Most common cancer stage: {df['6th Stage'].mode()[0]}")
    
    # Correlation between tumor size and survival months
    correlation = df['Tumor Size'].corr(df['Survival Months'])
    st.write(f"Correlation between tumor size and survival months: {correlation:.2f}")
    
    # Visualizations for Key Findings
    st.subheader("Survival Months by Cancer Stage")
    fig = px.box(df, x="6th Stage", y="Survival Months", color="Status",
                 title="Survival Months by Cancer Stage and Status")
    st.plotly_chart(fig)
    
    # Survival Rate by Cancer Stage
    stage_survival = df.groupby("6th Stage").agg({
        "Status": lambda x: (x == "Alive").mean() * 100
    }).reset_index()
    fig = px.bar(stage_survival, x="6th Stage", y="Status",
                 title="Survival Rate by Cancer Stage",
                 labels={"Status": "Survival Rate (%)"})
    st.plotly_chart(fig)

# Machine Learning
elif option == "Machine Learning":
    st.title("Machine Learning Analysis")
    
    # Prepare data
    X_train, X_test, y_train, y_test, preprocessor, label_encoder, numerical_features, categorical_features = prepare_data(df)
    
    # Train models
    with st.spinner('Training models...'):
        models = train_models(_X_train=X_train, _y_train=y_train, _preprocessor=preprocessor)

    # Model selection and evaluation section
    st.subheader("Model Evaluation")
    selected_model = st.selectbox("Select Model", list(models.keys()))
    
    if selected_model:
        model = models[selected_model]
        metrics = evaluate_model(model, X_test, y_test)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
        col2.metric("Precision", f"{metrics['Precision']:.3f}")
        col3.metric("Recall", f"{metrics['Recall']:.3f}")
        col4.metric("F1 Score", f"{metrics['F1 Score']:.3f}")
        
        # Confusion Matrix
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Create confusion matrix heatmap using plotly graph objects
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Survived', 'Deceased'],
            y=['Survived', 'Deceased'],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual"
        )
        st.plotly_chart(fig)

    # Feature importance section (for Random Forest)
    if selected_model == 'Random Forest':
        st.subheader("Feature Importance")
        
        feature_names = (numerical_features.tolist() + 
                        [f"{feat}_{val}" for feat, vals in 
                         zip(categorical_features, 
                             model.named_steps['preprocessor']
                             .named_transformers_['cat']
                             .named_steps['onehot']
                             .categories_) 
                         for val in vals[1:]])
        
        importances = model.named_steps['classifier'].feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True).tail(10)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                     title="Top 10 Feature Importances")
        st.plotly_chart(fig)

    # Prediction section
    st.subheader("Make Predictions")
    st.write("Enter patient information to predict survival status:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_age = st.number_input("Age", min_value=0, max_value=100, value=50)
        input_race = st.selectbox("Race", df['Race'].unique())
        input_marital = st.selectbox("Marital Status", df['Marital Status'].unique())
        input_t_stage = st.selectbox("T Stage ", df['T Stage '].unique())
        input_n_stage = st.selectbox("N Stage", df['N Stage'].unique())

        
    with col2:
        input_stage = st.selectbox("6th Stage", df['6th Stage'].unique())
        input_differentiate = st.selectbox("Differentiate", df['differentiate'].unique())
        input_grade = st.selectbox("Grade", df['Grade'].unique())
        input_a_stage = st.selectbox("A Stage", df['A Stage'].unique())
        input_tumor_size = st.number_input("Tumor Size", min_value=0, max_value=200, value=30)
        
    with col3:
        input_estrogen = st.selectbox("Estrogen Status", df['Estrogen Status'].unique())
        input_progesterone = st.selectbox("Progesterone Status", df['Progesterone Status'].unique())
        input_nodes_examined = st.number_input("Regional Nodes Examined", min_value=0, max_value=100, value=10)
        input_nodes_positive = st.number_input("Regional Nodes Positive", min_value=0, max_value=100, value=0)
        input_survival_months = st.number_input("Survival Months", min_value=0, max_value=200, value=12)

    # Create input dataframe
    input_data = pd.DataFrame({
        'Age': [input_age],
        'Race': [input_race],
        'Marital Status': [input_marital],
        'N Stage': [input_n_stage],
        '6th Stage': [input_stage],
        'differentiate': [input_differentiate],
        'Grade': [input_grade],
        'A Stage': [input_a_stage],
        'Tumor Size': [input_tumor_size],
        'Estrogen Status': [input_estrogen],
        'Progesterone Status': [input_progesterone],
        'Regional Node Examined': [input_nodes_examined],
        'Reginol Node Positive': [input_nodes_positive],
        'T Stage ': [input_t_stage],  # Added T Stage
        'Survival Months': [input_survival_months]  # Added Survival Months
    })

    if st.button("Predict"):
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        st.write("### Prediction Results")
        prediction_label = "Survived" if prediction[0] == 0 else "Deceased"
        survival_probability = probability[0][0] if prediction[0] == 0 else probability[0][1]
        
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.metric("Predicted Outcome", prediction_label)
        with result_col2:
            st.metric("Confidence", f"{survival_probability:.2%}")