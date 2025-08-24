import streamlit as st
import joblib
import numpy as np
import streamlit.components.v1 as components

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
                    # Get top 3 indices in descending order of probability
                    top_indices = np.argsort(probabilities)[-3:][::-1]
                    top_categories = model.classes_[top_indices]
                    top_probs = probabilities[top_indices] * 100  # Convert to percentage

                    # Ensure categories and probabilities are sorted in descending order
                    sorted_pairs = sorted(zip(top_probs, top_categories), reverse=True)
                    top_probs, top_categories = zip(*sorted_pairs) if sorted_pairs else ([], [])
                    top_probs = list(top_probs)
                    top_categories = list(top_categories)

                    st.subheader("Prediction:")
                    st.write(f"Predicted add-to-cart category: **{predicted_category}**")
                    st.write(f"Selected categories: {', '.join(viewed_categories_list)}")
                    st.subheader("Top 3 Predicted Categories:")
                    for cat, prob in zip(top_categories, top_probs):
                        st.write(f"- {cat}: {prob:.2f}% probability")

                    # Chart.js visualization with single-quoted f-string
                    chart_data = {
                        "labels": list(top_categories),
                        "probs": list(top_probs),
                        "selected": ', '.join(viewed_categories_list)
                    }
                    chart_html = f'<html>\
<head>\
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>\
</head>\
<body>\
    <canvas id="probChart" width="400" height="200"></canvas>\
    <script>\
        const ctx = document.getElementById("probChart").getContext("2d");\
        new Chart(ctx, {{ \
            type: "bar",\
            data: {{ \
                labels: {chart_data["labels"]},\
                datasets: [{{ \
                    label: "Prediction Probability (%)",\
                    data: {chart_data["probs"]},\
                    backgroundColor: ["#36A2EB", "#FFCE56", "#FF6384"],\
                    borderColor: ["#2E86C1", "#FFB300", "#E91E63"],\
                    borderWidth: 1\
                }}]\
            }},\
            options: {{ \
                indexAxis: "y",\
                responsive: true,\
                plugins: {{ \
                    tooltip: {{ \
                        callbacks: {{ \
                            footer: () => "Selected: {chart_data['selected']}"\
                        }}\
                    }},\
                    legend: {{ display: true }},\
                    title: {{ \
                        display: true,\
                        text: "Top Predicted Categories (Highest to Lowest)"\
                    }}\
                }},\
                scales: {{ \
                    x: {{ \
                        title: {{ display: true, text: "Probability (%)" }},\
                        max: 100\
                    }},\
                    y: {{ \
                        title: {{ display: true, text: "Category" }}\
                    }}\
                }}\
            }}\
        }});\
    </script>\
</body>\
</html>'
                    components.html(chart_html, height=300)
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}. Please check the input or model compatibility.")
else:
    st.warning("Model, vectorizer, or categories could not be loaded. Prediction is not available.")

st.write("Note: Predictions are based on viewing history and a trained Random Forest model.")