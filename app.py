import subprocess
import sys

# List of required libraries
required_libraries = [
    "streamlit",
    "pandas",
    "numpy",
    "plotly",
    "folium",
    "streamlit-folium",  # Add this line
    "scikit-learn"
]

# Function to install missing libraries
def install_library(library):
    subprocess.check_call([sys.executable, "-m", "pip", "install", library])

# Check and install missing libraries
for library in required_libraries:
    try:
        __import__(library)
    except ImportError:
        print(f"{library} not found. Installing...")
        install_library(library)

# Now import the libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static  # This should now work
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Rest of your code...

# Page Configuration
st.set_page_config(page_title="Real Estate Market Analysis", layout="wide")

# Title
st.title("Real Estate Market Analysis Dashboard")

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv("data/property_data.csv")  # Replace with your dataset
    return data

data = load_data()

# Sidebar Filters
st.sidebar.header("Filters")
region = st.sidebar.selectbox("Select Region", data['Region'].unique())
property_type = st.sidebar.selectbox("Select Property Type", data['PropertyType'].unique())
price_range = st.sidebar.slider(
    "Select Price Range", 
    float(data['Price'].min()), 
    float(data['Price'].max()), 
    (float(data['Price'].min()), float(data['Price'].max()))
)

# Filter Data
filtered_data = data[(data['Region'] == region) & (data['PropertyType'] == property_type) & (data['Price'].between(price_range[0], price_range[1]))]

# Display Filtered Data
st.write("### Filtered Data")
st.dataframe(filtered_data)

# Property Price Analysis
st.write("### Property Price Analysis")
fig = px.bar(filtered_data, x='Neighborhood', y='Price', color='PropertyType', title="Property Prices by Neighborhood")
st.plotly_chart(fig)

# Neighborhood Analysis
st.write("### Neighborhood Analysis")
avg_price = filtered_data.groupby('Neighborhood')['Price'].mean().reset_index()
fig2 = px.bar(avg_price, x='Neighborhood', y='Price', title="Average Property Prices by Neighborhood")
st.plotly_chart(fig2)

# Heatmap Visualization
st.write("### Heatmap of Property Prices")
map_center = [filtered_data['Latitude'].mean(), filtered_data['Longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=12)
heat_data = [[row['Latitude'], row['Longitude'], row['Price']] for index, row in filtered_data.iterrows()]
HeatMap(heat_data).add_to(m)
folium_static(m)

# Rental Yield Calculator
st.write("### Rental Yield Calculator")
price = st.number_input("Enter Property Price", min_value=0.0)
rental_income = st.number_input("Enter Monthly Rental Income", min_value=0.0)
if price > 0 and rental_income > 0:
    rental_yield = (rental_income * 12) / price * 100
    st.write(f"**Rental Yield:** {rental_yield:.2f}%")

# Demand Prediction (Simple Linear Regression)
st.write("### Demand Prediction")
if st.button("Train Demand Prediction Model"):
    X = filtered_data[['Price', 'Bedrooms', 'Bathrooms']]
    y = filtered_data['Demand']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    st.write("Model Trained Successfully!")
    st.write(f"R² Score: {model.score(X_test, y_test):.2f}")

# Export Data
if st.button("Export Filtered Data"):
    filtered_data.to_csv("filtered_data.csv", index=False)
    st.success("Data Exported Successfully!")