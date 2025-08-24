import streamlit as st
import joblib
import numpy as np

# Load the trained model and vectorizer
try:
    model = joblib.load('model/item_prediction_model.pkl')
    vectorizer = joblib.load('model/count_vectorizer.pkl')
    st.success("Model and vectorizer loaded successfully!")
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure 'item_prediction_model.pkl' and 'count_vectorizer.pkl' are in the same directory.")
    model, vectorizer = None, None
    all_categories = []  # Initialize as empty list
except Exception as e:
    st.error(f"Error loading model or vectorizer: {str(e)}")
    model, vectorizer = None, None
    all_categories = []  # Initialize as empty list
else:
    try:
        all_categories = vectorizer.get_feature_names_out().tolist()  # Convert to list for safety
        if not all_categories:
            st.warning("No categories found in the vectorizer. Please check the training data.")
    except Exception as e:
        st.error(f"Error retrieving categories from vectorizer: {str(e)}")
        all_categories = []

st.title("Item Category Prediction")
st.markdown("""
### About This App
This app predicts the category of an item a user is likely to add to their cart based on viewed categories.
For example, selecting 'electronics' and 'phones' predicts the most likely add-to-cart category.
""")

# Print output for all_categories
st.write(f"Kindly note: The number of categories loaded: {len(all_categories)}")

if model and vectorizer and all_categories:
    st.write("Select the category IDs of the items the user has viewed:")
    viewed_categories_list = st.multiselect("Viewed Categories", all_categories)

    if st.button("Predict Add-to-Cart Category"):
        if not viewed_categories_list:
            st.warning("Please select at least one viewed category.")
        else:
            try:
                feature_str = ' '.join(viewed_categories_list)
                X_predict = vectorizer.transform([feature_str])
                if X_predict.nnz == 0:
                    st.warning("Selected categories do not match known features. Try different categories.")
                else:
                    predicted_category = model.predict(X_predict)[0]
                    probabilities = model.predict_proba(X_predict)[0]
                    top_indices = np.argsort(probabilities)[-3:][::-1]
                    top_categories = model.classes_[top_indices]
                    top_probs = probabilities[top_indices]

                    st.subheader("Prediction:")
                    st.write(f"Predicted add-to-cart category: **{predicted_category}**")
                    st.subheader("Top 3 Predicted Categories:")
                    for cat, prob in zip(top_categories, top_probs):
                        st.write(f"- {cat}: {prob:.2%} probability")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}. Please check the input or model compatibility.")
else:
    st.warning("Model, vectorizer, or categories could not be loaded. Prediction is not available.")

st.write("Note: Predictions are based on viewing history and a trained Random Forest model.")