import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

# Clear Streamlit cache to ensure fresh rendering
st.cache_data.clear()

# Load the trained model and vectorizer
try:
    model = joblib.load('model/item_prediction_model.pkl')
    vectorizer = joblib.load('model/count_vectorizer.pkl')
    st.success("Model and vectorizer loaded successfully!")
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure 'item_prediction_model.pkl' and 'count_vectorizer.pkl' are in the model directory.")
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

# Debugging output for categories
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

                    st.subheader("Prediction:")
                    st.write(f"Predicted add-to-cart category: **{predicted_category}**")
                    st.write(f"Selected categories: {', '.join(viewed_categories_list)}")

                    # Display top 3 probabilities as percentages
                    st.subheader("Top 3 Predicted Categories:")
                    class_labels = model.classes_
                    prob_df = pd.DataFrame({'Category': class_labels, 'Probability': probabilities * 100})  # Convert to percentage
                    prob_df = prob_df.sort_values('Probability', ascending=False).head(3)  # Top 3
                    for _, row in prob_df.iterrows():
                        st.write(f"- {row['Category']}: {row['Probability']:.2f}% probability")

                    # Create vertical bar chart
                    plt.figure(figsize=(10, 6))
                    ax = sns.barplot(x='Category', y='Probability', data=prob_df, palette='viridis', order=prob_df['Category'])
                    plt.title(f"Top 3 Predicted Category Probabilities: {', '.join(prob_df['Category'])}")
                    plt.xlabel("Category ID")
                    plt.ylim(0, 100)  # Set y-axis range to 0-100%

                    # Format y-axis as percentages (redundant since y-axis is hidden, but included for robustness)
                    formatter = mticker.PercentFormatter(xmax=100.0)
                    ax.yaxis.set_major_formatter(formatter)

                    # Add data labels
                    for container in ax.containers:
                        labels = [f'{h:.1f}%' if (h := v.get_height()) > 0 else '' for v in container]
                        ax.bar_label(container, labels=labels, label_type='edge', fontsize=10)

                    # Remove y-axis
                    ax.yaxis.set_visible(False)
                    ax.tick_params(axis='y', which='both', length=0)  # Remove y-axis ticks

                    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
                    plt.tight_layout()
                    st.pyplot(plt)
                    plt.clf()  # Clear figure to prevent overlap
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}. Please check the input or model compatibility.")
else:
    st.warning("Model, vectorizer, or categories could not be loaded. Prediction is not available.")

st.write("Note: Predictions are based on viewing history and a trained Random Forest model.")