import streamlit as st
import joblib
import numpy as np
import pandas as pd
from collections import Counter

# Load the model and vectorizer
try:
    model = joblib.load('models/random_forest_model.pkl')
    vectorizer = joblib.load('models/count_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please check the file paths.")
    st.stop()

st.title('Item Property Prediction')

st.write("Enter the sequence of viewed category IDs (space-separated) to predict the category of the item that will be added to the cart.")

# Input for viewed category IDs
viewed_categories_input = st.text_input("Viewed Category IDs (space-separated)")

if st.button('Predict Category'):
    if viewed_categories_input:
        try:
            # Prepare the input feature
            input_feature = ' '.join(viewed_categories_input.split())
            X_input = vectorizer.transform([input_feature])

            # Predict the category
            predicted_category = model.predict(X_input)

            st.success(f"Predicted Category ID for Add to Cart: {predicted_category[0]}")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
    else:
        st.warning("Please enter viewed category IDs.")

st.write("---")
st.write("This application uses a trained model to predict the category of an item added to the cart based on the sequence of previously viewed item categories by a visitor.")