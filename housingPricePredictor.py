import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing
import joblib  # To save/load the model
import matplotlib.pyplot as plt
import seaborn as sns

# Cache the dataset loading process (only load it once)
@st.cache_data
def load_data():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target
    return X, y

# Cache the model training process (train only once)
@st.cache_resource
def train_model(X, y):
    # Split into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the RandomForestRegressor model
    rf_model = RandomForestRegressor(
        n_estimators=200,       # Number of trees in the forest
        max_depth=20,           # Maximum depth of the trees
        min_samples_split=5,    # Minimum number of samples required to split an internal node
        min_samples_leaf=2,     # Minimum number of samples required to be at a leaf node
        max_features='sqrt',    # Number of features to consider for the best split
        bootstrap=True,         # Whether bootstrap samples are used when building trees
        random_state=42         # For reproducibility
    )
    
    rf_model.fit(X_train_scaled, y_train)

    return rf_model, scaler, X_test_scaled, y_test

# Load data
X, y = load_data()

# Train the model and get necessary objects
rf_model, scaler, X_test_scaled, y_test = train_model(X, y)

# Model Evaluation
y_pred = rf_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit App
st.title("California Housing Price Prediction")
st.write("This app predicts house prices in California based on various features.")

# Allow user input for prediction
st.sidebar.header("User Input")
user_input = {
    'MedInc': st.sidebar.slider("Median Income (in 1000s)", min_value=0.5, max_value=15.0, value=8.3, step=0.1),
    'HouseAge': st.sidebar.slider("House Age (in years)", min_value=1, max_value=100, value=40),
    'AveRooms': st.sidebar.slider("Average Rooms", min_value=1, max_value=10, value=6),
    'AveBedrms': st.sidebar.slider("Average Bedrooms", min_value=1, max_value=5, value=2),
    'Population': st.sidebar.slider("Population", min_value=100, max_value=5000, value=1500),
    'AveOccup': st.sidebar.slider("Average Occupancy", min_value=1.0, max_value=5.0, value=2.5),
    'Latitude': st.sidebar.slider("Latitude", min_value=32.0, max_value=42.0, value=37.0),
    'Longitude': st.sidebar.slider("Longitude", min_value=-125.0, max_value=-115.0, value=-120.0)
}

# Convert user input into a dataframe
user_data = pd.DataFrame(user_input, index=[0])

# Standardize the user input
user_data_scaled = scaler.transform(user_data)

# Predict the house price for user input
prediction = rf_model.predict(user_data_scaled)
st.subheader(f"Predicted House Price: ${prediction[0] * 1000:,.2f}")

# Show model evaluation results
st.write(f"### Model Evaluation on Test Set")
st.write(f"**Mean Absolute Error (MAE):** {mae:.3f}")
st.write(f"**R-squared (R2):** {r2:.3f}")

# Feature Importance Visualization
st.write("### Feature Importance")
importances = rf_model.feature_importances_
feature_names = X.columns

# Create a DataFrame of feature importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax, color='skyblue')
st.pyplot(fig)
