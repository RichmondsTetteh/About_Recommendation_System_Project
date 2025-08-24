# Model Folder

The model folder contains machine learning models and related resources for the Recommendation System Project.

The trained model and the vectorizer were saved: The trained RandomForestClassifier model was finally saved to a file named random_forest_model.pkl using joblib. The fitted CountVectorizer was also saved to a file named count_vectorizer.pkl. This is important because I would need to use the same vectorizer to transform new data before making predictions with the saved model. Messages were printed to confirm the saving of the model and vectorizer files.

## Files
- `item_prediction_model.pkl`
- `count_vectorizer.pkl`

These files will be used for making item predictions.

**App Name: Item Category Prediction**

The app predicts the category of an item a user is likely to add to their cart based on viewed categories. For example, selecting 'electronics' and 'phones' predicts the most likely add-to-cart category.
The app then displays the top 3 items to be added to the cart while displaying their probabilities.
