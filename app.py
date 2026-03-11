import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

# Title
st.title("🌸 Iris Flower Classification Dashboard")
st.markdown("Predict Iris flower species using Machine Learning")

# Load dataset
@st.cache_resource
def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    df = pd.DataFrame(X, columns=iris.feature_names)
    df["species"] = y
    return df, iris

df, iris = load_data()

# Train model
@st.cache_resource
def train_model():
    model = DecisionTreeClassifier()
    model.fit(iris.data, iris.target)
    return model

model = train_model()

# Sidebar inputs
st.sidebar.header("🌼 Enter Flower Measurements")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Layout columns
col1, col2 = st.columns(2)

# Prediction section
with col1:

    st.subheader("🔍 Prediction")

    if st.button("Predict Flower Species"):

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        species = iris.target_names[prediction][0]

        st.success(f"Predicted Species: **{species}**")

        st.subheader("Prediction Probability")

        prob_df = pd.DataFrame(
            probability,
            columns=iris.target_names
        )

        st.bar_chart(prob_df.T)

# Dataset preview
with col2:

    st.subheader("📊 Iris Dataset Preview")
    st.dataframe(df.head())

# Visualization section
st.markdown("---")
st.subheader("📈 Feature Visualization")

feature = st.selectbox(
    "Select Feature",
    iris.feature_names
)

fig, ax = plt.subplots()

for i, species in enumerate(iris.target_names):
    ax.hist(
        df[df["species"] == i][feature],
        label=species,
        alpha=0.6
    )

ax.set_xlabel(feature)
ax.set_ylabel("Count")
ax.legend()

st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown(
"""
### ℹ About this App
This application uses a **Decision Tree Machine Learning model** trained on the **Iris dataset** to classify flowers into:

- Setosa  
- Versicolor  
- Virginica
"""
)