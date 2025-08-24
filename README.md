# About_Recommendation_System_Project
This project focuses on processing, analyzing and training a dataset related to a recommendation system from an e-commerce site.

The notebook begins by importing the necessary libraries. Pandas was imported for data manipulation and numpy was imported for numerical operations.

The read_in_chunks function was defined to efficiently read large CSV files by splitting them into smaller chunks, which helps manage memory usage.

The following code specified the file paths for events.csv and category_tree.csv stored in my Google Drive.
The events.csv file was then read in chunks using the read_in_chunks function, and these chunks were concatenated into a single DataFrame named events.

Similarly, the category_tree.csv file was read in chunks and concatenated into a DataFrame called category_tree.

Following the data loading, the notebook proceeds with data preprocessing for the events DataFrame:
It checked for and reports the number of duplicate rows in the events DataFrame. The output showed 460 duplicate rows. These duplicate rows are then removed.
A subsequent check verified that the duplicates have been successfully removed, showing 0 duplicate rows.
The code then checked for missing values (NA) and empty strings in the events DataFrame. The output indicated 2733184 missing values in the 'transactionid' column and no empty strings.
Finally, the data types of the columns in the events DataFrame are displayed, showing timestamp, visitorid, and itemid as int64, event as object, and transactionid as float64.

The code checks for duplicate rows, and the output confirms 0 duplicate rows in the Category_tree DataFrame.
It then checks for missing values and empty strings. The output showed 25 missing values in the 'parentid' column and no empty strings.
The missing values in the 'parentid' column are replaced with the median value of the existing 'parentid' values, The NA values were ignored during the median calculation.
A check after handling NAs confirms 0 missing values in the category_tree DataFrame.
Finally, the data types for category_tree were displayed, showing categoryid as int64 and parentid as float64.

The notebook continues by cleaning the events data:

It replaced the missing values in the transactionid column with 0. This is done because 'view' and 'addtocart' events do not involve a transaction, so a missing transaction ID can be interpreted as no transaction occurring. The output confirms 0 missing values in transactionid after filling.
The visitorid and itemid columns were converted to string type.
The event column was converted to the 'category' data type to optimize memory usage and speed up operations on this column, as it contained a limited number of unique values.
The data type of the timestamp column was checked before and after conversion. Initially, it was an int64, and after conversion, it became datetime64[ns, UTC], indicating a successful conversion from milliseconds since epoch to a timezone-aware datetime object.

Next, the notebook visualized the distribution of user events:

The first step was to calculate the count of each event type (view, addtocart, transaction).
A bar plot was generated to visualize these counts. The output showed a bar chart titled "Distribution of User Events" with the counts for each event type. It was observed that the 'view' events are by far the most frequent, followed by 'addtocart' and then 'transaction' events.

The notebook then analyzed event counts and conversion rates per visitor:

It grouped the events by visitorid and event to count how many of each event type each visitor performed. The unstack(fill_value=0) operation reshaped the data so that each event type became a column, and missing values were filled with 0.
The columns were renamed to num_view, num_addtocart, and num_transaction for clarity.
The result displayed, showing a table with visitorid and the counts for each event type for the first few visitors.

Conversion rates are calculated for each visitor:
conversion_view_to_add: Number of 'addtocart' events divided by the number of 'view' events.
conversion_add_to_transaction: Number of 'transaction' events divided by the number of 'addtocart' events.
conversion_view_to_transaction: Number of 'transaction' events divided by the number of 'view' events.
These new conversion rate columns were added to the user_event_counts DataFrame, and the first few rows were displayed, showing the calculated conversion rates per visitor.

Finally, a histogram was generated for the view-to-add conversion rate:

A histogram was plotted to show the distribution of the conversion_view_to_add values across all visitors. The output shows a histogram titled "Distribution of View-to-Add Conversion Rate". The histogram displayed was heavily skewed towards 0, indicating that most users who viewed items did not add them to their cart.
This section effectively cleaned the event data and provides initial visualizations of event distributions and conversion rates at a per-user level.

Anomaly detection and bot removal

This section of the notebook aimed to identify and remove potential bots or abnormal user behavior from the events data.

First, the events DataFrame was sorted by visitorid and timestamp. This is crucial for calculating the time difference between consecutive events for each visitor.
A new column time_diff was created, which calculated the time difference in seconds between consecutive events for each visitor. The groupby('visitorid')['timestamp'].diff().dt.total_seconds() part calculated the difference within each visitor group. The output displayed the first few rows of the events DataFrame with the new time_diff column. There was NaN for the first event of each visitor, as there is no previous event to calculate the difference from.
The next step was to aggregate the statistics per visitor:
It grouped the data by visitorid and calculated:

total_events: The total number of events for each visitor.
avg_time_diff: The average time difference between events for each visitor.
median_time_diff: The median time difference between events for each visitor.

The output showed the visitor_stats DataFrame with these aggregated metrics for the first few visitors.

The potential bots are then identified using a rule-based approach.

Thresholds were calculated for total_events (95th percentile) and avg_time_diff (5th percentile). The idea is that users with an unusually high number of events or very short time gaps between events might be bots.
A bot_flag column was created in visitor_stats, set to True if a user's total_events was above the calculated threshold or their avg_time_diff was below the threshold.
The potential bots flagged by this rule are then inspected. The output showed a table of users flagged as potential bots and the number of such users (58627 potential bots). The top ten bots by total events were also displayed.
The notebook then employed Isolation Forest, an unsupervised machine learning algorithm, for anomaly detection. 

The Features (total_events, avg_time_diff, and median_time_diff) are selected from the visitor_stats DataFrame. The data types of these features are checked to ensure they are numeric. An Isolation Forest model is then initialized and fitted to these features.
Anomaly scores are calculated for each user using the fitted model. Lower scores indicate a higher degree of anomaly. The type, shape, and first few elements of the anomaly_scores are printed for debugging.
The anomaly_scores were added as a new column to the visitor_stats DataFrame. A check was included to ensure the lengths of the scores and the DataFrame match.
An anomaly threshold was defined as the 99th percentile of the anomaly scores. Users with a score above this threshold were flagged as anomalies with -1 in the anomaly_flag column, while others are flagged as 1.
The updated visitor_stats DataFrame with the anomaly_score and anomaly_flag columns was displayed.
Finally, the two detection criteria (rule-based and Isolation Forest) were combined to create a final bot flag:

The thresholds for the rule-based detection were recalculated.
A bot_flag_rule column was created based on these thresholds.
A final_bot_flag column was created, set to True if either the bot_flag_rule is True or the anomaly_flag is -1.
The number of users flagged as bots by this combined approach was printed (58627 bots detected), which is the same number as the rule-based detection in this case.
The events associated with these flagged bot users are then removed from the events DataFrame, resulting in a cleaned_events DataFrame. The number of unique users remaining after bot removal was printed (1348953 unique users), indicating a significant reduction in the number of users.

After the bot removal, several visualizations were generated using the cleaned_events data:

A scatter plot of total_events versus avg_time_diff per user was created. The x-axis is on a log scale due to the wide range of total events. The output showed this scatter plot, which can help visualize the distribution of user activity and time differences.

A bar plot of the distribution of user events was generated again using the cleaned_events data. The output showed a bar chart similar to the earlier one but with potentially lower counts for each event type after removing bot activity. The y-axis limit is set to 3,000,000 for better visualization after bot removal.

The top 10 most viewed products were identified and displayed as a bar plot. This was done by filtering for 'view' events in cleaned_events, counting views per itemid, and selected the top 10. The output showed the table of the top 10 viewed products and a bar chart visualized their view counts.

Similarly, the top 10 most purchased products were identified and displayed as a bar plot by filtering for 'transaction' events in cleaned_events, counting purchases per itemid, and selected the top 10. The output showed the table of top 10 purchased products and a bar chart visualized their purchase counts.

A conversion funnel was visualized by aggregating event counts for 'view', 'addtocart', and 'transaction' events in the cleaned_events data. The output showed a table with the counts for each event type and a bar plot represented the conversion funnel.

The time series plots of daily and monthly total events were generated from the cleaned_events data. The timestamp was used to extract the date and month, and events are aggregated accordingly. The outputs show tables with daily and monthly event counts and line plots visualizing the trend of total events over time.

The relationship between overall activity (total_events) and conversion rate (conversion_view_to_add) was visualized using a scatter plot on the visitor_stats_cleaned DataFrame (derived from cleaned_events). The output showed the scatter plot with total events on a log scale.

A heatmap of event activity by hour and day of the week was generated from the cleaned_events data. The timestamp was used to extract the hour and weekday, and event counts are aggregated. The output showed a heatmap illustrating the distribution of events across different hours and weekdays.

After the visualization the notebook handled the item_properties_part1.csv and item_properties_part2.csv files, which contain information about various properties of the items.

The notebook specified the file paths for item_properties_part1.1.csv and item_properties_part2.csv in my Google Drive.
Both files were read in chunks using the read_in_chunks function, similar to the events data. This was done to handle potentially large files efficiently.
The chunks from item_properties_part1.1.csv were concatenated into a DataFrame named item_properties_1.
The chunks from item_properties_part2.csv were concatenated into a DataFrame named item_properties_2.
The two DataFrames (item_properties_1 and item_properties_2) were then combined into a single DataFrame called item_properties.
The first few rows of the item_properties DataFrame were displayed to verify the structure and content. The output showed columns like timestamp, itemid, property, and value.

The notebook then performed data cleaning on the item_properties data:

Checked for duplicate rows in the item_properties DataFrame. The output confirmed 0 duplicate rows.
Checked for missing values (NA) and empty strings. The output showed 0 missing values and 0 empty strings in all columns.
The timestamp column was converted from milliseconds since epoch to a timezone-aware datetime object, similar to the events data. The output confirmed the data type of timestamp after conversion was datetime64[ns, UTC].

After cleaning the individual datasets, the notebook proceeded to merge the cleaned_events DataFrame with the item_properties DataFrame. To avoid potential issues with chained assignment (SettingWithCopyWarning), a copy of cleaned_events was created. The itemid columns in both cleaned_events and item_properties were converted to string type to ensure consistent data types for merging.
A left join was performed to merge cleaned_events with item_properties based on the itemid column. This means that all events from cleaned_events are kept, and matching item properties are added. Suffixes (_event and _prop) were added to the timestamp columns to distinguish them.
The merged DataFrame was then filtered to keep only the property entries where the property timestamp is less than or equal to the event timestamp. This ensures that I consider only item properties that were recorded before or at the time of the event.
For each event, the most recent property entry (based on property_timestamp) was selected using groupby and idxmax(). This was done to associate each event with the latest available information about the item's properties at that time.
Columns were renamed for clarity, and the relevant columns are selected and reordered to form the events_items DataFrame.

The notebook then addressed the 'categoryid' property:
It replaced the value in the property column with the corresponding value from the value column for rows where the property was 'categoryid'. This effectively moved the actual category ID from the value column to the property column when the property type was 'categoryid'.
A sample of the events_items DataFrame was displayed to verify the changes. The output showed sample rows with the updated 'property' column.
Next, the events_items DataFrame was merged with the category_tree DataFrame:

The data types of the property column in events_items and the categoryid column in category_tree were checked and converted to string type to ensure a successful merge.
A left join was performed to merge events_items with category_tree based on the property column in events_items and the categoryid column in category_tree. This added the parentid information from the category_tree to the events_items DataFrame where the 'property' in events_items corresponds to a categoryid.
The head of the merged DataFrame was displayed, showing the added categoryid and parentid columns.

The notebook then focused on filtering and handling the 'available' property:

Filtered the item_properties DataFrame to keep only rows where the property was 'available'.
For each itemid, it kept only the most recent 'available' property entry.
The value column (which indicates availability, likely 0 or 1) was converted to an integer type.
Only the itemid and available columns were kept in this filtered DataFrame.
The latest availability status per item was then merged into the merged_df based on the itemid.
Printed to verify the columns and number of rows in merged_df after adding the 'available' status, and a sample was displayed.
The missing values in the 'available' column were filled with 0 and converted to integer type.
Duplicate columns in merged_df were checked and removed, although the output indicated no duplicate columns were found.

The notebook then prepared the final_df:

Checked the unique values in the categoryid column of merged_df. The output showed the counts of different categoryid values, including NaN.
A new DataFrame final_df is created by removing rows where the categoryid is 'NaN'.
Finally, missing values in final_df were handled:
Missing values in the available column were replaced with 0.
Missing values in the parentid column were replaced with -1.
The number of missing values in final_df is checked after these steps. The output showed 227388 missing values remaining in the categoryid column, even though the previous step aimed to remove rows with 'NaN' in 'categoryid'. This suggested that the check for 'NaN' as a string might not have removed actual NaN values.
The code explicitly drops rows with missing values in the categoryid column using dropna().
A final check confirms that there are 0 missing values in all columns of the final_df.

This section of the notebook explored how item availability impacts user interactions, how conversion rates vary across different times of the day, and the relationship between user session duration and purchase likelihood, and the distribution of event types.

First, the notebook investigated the impact of item availability on user interactions:

It aggregated event counts by availability status (0 for unavailable, 1 for available) from the final_df. Rows with missing values in the 'available' column were excluded from this aggregation.
The output displayed a table showing the counts of 'addtocart', 'transaction', and 'view' events for both available and unavailable items. As seen in the output, there are significantly more events for available items, particularly 'view' events.
A bar plot was generated to visualize these event counts by availability status. The output showed a bar chart titled "Impact of Item Availability on User Interactions" with separate bars for available (green) and unavailable (red) items for each event type. This visualization clearly showed that available items have a much higher number of all event types compared to unavailable items, highlighting the importance of item availability for user engagement.

The notebook then examined how conversion rates vary across different times of the day:

A copy of final_df was created to avoid potential warnings.
The hour of the day was extracted from the timestamp column and added as a new column hour.
Conversion rates were aggregated by hour. Two conversion rates were calculated for each hour:
conversion_view_to_add: Number of 'addtocart' events divided by the number of 'view' events for that hour.
conversion_add_to_purchase: Number of 'transaction' events divided by the number of 'addtocart' events for that hour.
The output displayed a table showing these hourly conversion rates.

The data was then reshaped into a long format suitable for plotting multiple conversion rate types on the same graph.
A line plot was generated to visualize the conversion rates over the hours of the day. The output showed a line graph titled "Conversion Rates by Hour of the Day". There are two lines, one for 'conversion_view_to_add' and one for 'conversion_add_to_purchase'. The x-axis represents the hour of the day. This plot allows to observe any patterns or peaks in conversion rates throughout the day.
The notebook then investigates the relationship between user session duration and purchase likelihood:

Session statistics are calculated per user from the cleaned_events data (note that this part uses cleaned_events instead of final_df). This includes:
session_duration: The time difference between the first and last event for each user in minutes.
total_transactions: The total number of transaction events for each user.
total_views: The total number of view events for each user.

A conversion_rate (total transactions / total views) is calculated for each user, handling cases with zero views by setting the conversion rate to NaN.
Users with NaN conversion rates or session durations are filtered out.
The output displayed the session_stats DataFrame with these calculated metrics for the first few users.
Session durations are binned into categories (<5, 5-10, 10-20, 20-30, 30+ minutes). The output showed the session_stats DataFrame with the added duration_bin column.
The distribution of users across these duration bins is checked using value_counts().
The average conversion rate was summarized for each duration bin. The output showed a table with the average conversion rate and the count of users in each bin.
A bar chart was generated to visualize the average conversion rate by session duration bin. The output showed a bar plot titled "Average Conversion Rate by Session Duration". The x-axis represents the session duration bins, and the y-axis represents the average conversion rate. This plot helps understand if longer or shorter sessions are associated with higher conversion rates.

Finally, the notebook visualized the distribution of event types across all users using the final_df:
It aggregated the total counts for each event type ('view', 'addtocart', 'transaction') from the final_df.
The overall sum of events was calculated.
The proportion of each event type was calculated by dividing the count of each event type by the overall total.
A bar chart was generated to visualize these proportions. The y-axis is formatted as percentages. The output showed a bar chart titled "Distribution of Event Types Across Users", illustrating the percentage of each event type in the dataset. Similar to the earlier plot, this showed that 'view' events make up the vast majority of events.
This section provides valuable insights into user behavior by analyzing the impact of item availability, the temporal patterns of conversion, the relationship between session duration and purchase likelihood, and the overall distribution of event types.

The next section aimed to build an algorithm to predict the category of an item added to the cart based on the items viewed by the same visitor before the "add to cart" event.

The notebook starts by checking for missing values in final_df_copy, which was used in the previous section for visualizations. The output confirms there are no missing values in this DataFrame. The head and info of final_df_copy are also displayed, showing the columns and their data types.
The necessary libraries for modeling were imported 
train_test_split was imported for splitting data
CountVectorizer was imported for feature extraction
RandomForestClassifier was imported for the model
Accuracy_score and classification_report was imported for evaluation
Counter was impored for counting elements.

A copy of final_df_copy was made and assigned to df.
The DataFrame df was filtered to include only 'view' and 'addtocart' events, as these are the events relevant to the prediction task.

The data was prepared for prediction:
The code iterated through each visitor's events, grouped by visitorid.
For each visitor, the events were sorted by timestamp.
It identified the indices of 'addtocart' events.
For each 'addtocart' event, it extracted all 'view' events that occurred before that specific 'addtocart' event.
If there were no prior 'view' events, the 'addtocart' event was skipped for prediction.
The categoryid values from these prior 'view' events were collected as a list of strings.
A simple feature representation was created. A single string where the viewed category IDs were joined by spaces. This is a "bag of categories" approach.
**The target variable is the categoryid of the item in the current 'addtocart' event.**
A list of dictionaries data was populated with these feature strings and target category IDs.
The collected data is then converted into a pandas DataFrame pred_df.

A check was performed to see if pred_df was empty after preparing the data.
If pred_df was not empty, the notebook optionally filters the data to include only the top N most frequent categories in the target variable. This can help manage the complexity of the model by focusing on the most common categories. The threshold top_n was set to 100. A message was printed if no data remained after this filtering.

If there was data after filtering then Feature Vectorization: CountVectorizer is used to convert the 'bag of categories' strings into a numerical feature matrix (X). CountVectorizer creates a vocabulary of unique categories and represents each feature string as a vector where each element is the count of a specific category in the string.
**The target variable y was set to the 'target' column of pred_df.**

The data was split into training and testing sets:
train_test_split divides the feature matrix X and target vector y into training and testing sets (X_train, X_test, y_train, y_test). The indices of the original pred_df are also split (train_idx, test_idx).
The sparse matrices X_train and X_test were converted to dense DataFrames, keeping the original indices.

A Random Forest Classifier model was trained:
A RandomForestClassifier was initialized with 100 estimators and a random state for reproducibility. class_weight='balanced' was used to handle potential class imbalance in the target categories.
The model was fitted to the training data (X_train, y_train).
The model's performance was then evaluated

Predictions (y_pred) were made on the test set (X_test).
The accuracy of the model was calculated and printed. The output showed an Accuracy of 0.8568.
A classification report was printed, providing metrics like precision, recall, and f1-score for each category, as well as overall averages. The report showed varying performance across different categories, with some having high precision and recall, while others (especially those with fewer examples) performed poorly.

A baseline model was calculated for comparison:
A simple baseline was implemented for each test case and predicted the most frequent category among the items viewed before the add-to-cart event.
The accuracy of this baseline model was calculated and printed. The output showed a Baseline Accuracy (Most Frequent Category) of 0.8298. Comparing this to the Random Forest model's accuracy (0.8568) indicates that the Random Forest model provides a modest improvement over simply predicting the most viewed category.

A confusion matrix was visualized for the top N categories:
To make the confusion matrix readable, only the top 20 categories based on their frequency in the test set were generated.
confusion_matrix was calculated, and a heatmap was generated using seaborn. The output showed a heatmap representing the confusion matrix, with true labels on the y-axis and predicted labels on the x-axis. The numbers in the cells indicate the count of instances where a true category was predicted as a specific category. This visualization helps understand which categories are being predicted correctly and where the model is making mistakes (e.g., confusing similar categories).

Finally, the trained model and the vectorizer were saved:
The trained RandomForestClassifier model was initially saved to a file named random_forest_model.joblib using joblib.
The fitted CountVectorizer was also saved to a file named count_vectorizer.joblib. This is important because I would need to use the same vectorizer to transform new data before making predictions with the saved model. Messages were printed to confirm the saving of the model and vectorizer files.

I quantified and visualized the impact of removing the flagged bot users on key metrics and analyze the characteristics of the identified bots.

I analyzed the characteristics of flagged bots. I focused on the distribution of total events and average time difference specifically for the flagged bot users.
I checked if bot_users DataFrame is empty: It first checked if the bot_users DataFrame (which contains the statistics for users flagged as bots) is not empty. If it was empty, it printed a message indicating that bot users were not found and the analysis cannot proceed.
I printed Descriptive Statistics: If bot_users was not empty, it printed descriptive statistics (.describe()) for the 'total_events' and 'avg_time_diff' columns of the bot_users DataFrame. This gave a summary of the event counts and time differences for the flagged bots (e.g., mean, standard deviation, min, max, quartiles).
I visualized the Distributions: I created a figure with two subplots to visualize the distributions of 'total_events' and 'avg_time_diff' for the bot users using histograms (sns.histplot).
1.	Distribution of Total Events for Bots: The first histogram shows the distribution of the total number of events for each flagged bot. A log scale was used for the y-axis (plt.yscale('log')) because there might be a wide range of total events, with some bots having a significantly higher number than others.
2.	Distribution of Average Time Difference for Bots: The second histogram showed the distribution of the average time difference between events for the flagged bots. Again, a log scale was used for the y-axis for similar reasons.

Exported necessary files for app development

The trained model and the vectorizer were saved: The trained RandomForestClassifier model was finally saved to a file named random_forest_model.pkl using joblib. The fitted CountVectorizer was also saved to a file named count_vectorizer.pkl. 
