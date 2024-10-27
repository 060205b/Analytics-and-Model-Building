# Import required libraries: pandas, numpy, matplotlib, seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Load dataset and display first 10 rows
data = pd.read_csv('Dataset.csv')
data.head(10)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# List all column names
data.columns.tolist()

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Count missing values in each column
data.isnull().sum()

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Count unique values in 'Has Table booking'
data['Has Table booking'].value_counts()

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Count unique values in 'Has Online delivery'
data['Has Online delivery'].value_counts()

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Calculate and display the percentage of table booking and online delivery
number_of_restaurant = len(data)
table_booking = data[data['Has Table booking'] == 'Yes'].shape[0]
online_delivery = data[data['Has Online delivery'] == 'Yes'].shape[0]
table_booking_perct = (table_booking / number_of_restaurant) * 100
online_delivery_perct = (online_delivery / number_of_restaurant) * 100

print(f"Percentage of restaurants offering table booking: {table_booking_perct:.2f}%")
print(f"Percentage of restaurants offering online delivery: {online_delivery_perct:.2f}%")

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# View unique values of 'Aggregate rating'
data['Aggregate rating'].unique()

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plot the distribution of 'Aggregate rating' based on 'Has Table booking'
plt.figure(figsize=(12, 6))
sns.histplot(data=data, x='Aggregate rating', hue='Has Table booking', multiple='stack', palette='viridis', kde=True, bins=10)
plt.xlabel('Has Table Booking')
plt.ylabel('Average Aggregate Rating')
plt.title('Average Ratings of Restaurants with and without Table Booking')
plt.show()

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Calculate average ratings for restaurants with and without table booking
avg_ratings = data.groupby('Has Table booking')['Aggregate rating'].mean().reset_index()
avg_rating_with_booking = avg_ratings.loc[avg_ratings['Has Table booking'] == 'Yes', 'Aggregate rating'].values[0]
avg_rating_without_booking = avg_ratings.loc[avg_ratings['Has Table booking'] == 'No', 'Aggregate rating'].values[0]

print("Restaurants with table booked:", avg_rating_with_booking)
print("Restaurants without table booked:", avg_rating_without_booking)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plot distribution of ratings for restaurants with table booking
with_table_booking = data[data['Has Table booking'] == 'Yes']
plt.figure(figsize=(12, 6))
sns.histplot(with_table_booking['Aggregate rating'], bins=10, kde=False)
plt.xlabel('Aggregate Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Ratings for Restaurants with Table Booking')
plt.show()

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plot distribution of ratings for restaurants without table booking
without_table_booking = data[data['Has Table booking'] == 'No']
plt.figure(figsize=(12, 6))
sns.histplot(without_table_booking['Aggregate rating'], bins=10, kde=False)
plt.xlabel('Aggregate Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Ratings for Restaurants without Table Booking')
plt.show()

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Count unique values in 'Has Online delivery' again for verification
data['Has Online delivery'].value_counts()

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plot count of price range by online delivery availability
plt.figure(figsize=(10, 6))
sns.countplot(x='Price range', hue='Has Online delivery', data=data, palette='viridis')
plt.xlabel('Price Range')
plt.ylabel('Number of Restaurants')
plt.title('Availability of Online Delivery Among Restaurants with Different Price Ranges')
plt.xticks(rotation=45)
plt.legend(title='Online Delivery', labels=['No', 'Yes'])
plt.show()

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plot count of price range specifically for restaurants with online delivery
restaurants_with_online_deli = data[data['Has Online delivery'] == 'Yes']
plt.figure(figsize=(20, 10))
sns.countplot(x='Price range', data=restaurants_with_online_deli, palette='viridis')
plt.xlabel('Price Range')
plt.ylabel('Number of Restaurants with Online Delivery')
plt.title('Availability of Online Delivery Among Restaurants with Different Price Ranges')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# View unique price ranges
data['Price range'].unique()

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Calculate and display the most common price range
common_price_range = data['Price range'].mode()
print("The most common price range among all the restaurants is:", common_price_range)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Calculate average rating for each price range
avg_rating_price_range = data.groupby("Price range")['Aggregate rating'].mean().reset_index()
dataframe = pd.DataFrame(avg_rating_price_range)
print(dataframe)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Calculate average rating for each price range and add rating color
avg_rating_price_range.columns = ['Price range', 'Avg Rating']
unique_price_ranges = data.drop_duplicates(subset=['Price range'])
avg_rating_price_range = pd.merge(avg_rating_price_range, unique_price_ranges[['Price range', 'Rating color']], on='Price range', how='left')
dataframe = pd.DataFrame(avg_rating_price_range)
print(dataframe)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Identify the rating color of the highest average rating
highest_average_rating = data.loc[data['Aggregate rating'].idxmax()]
color_of_avg_rating = highest_average_rating['Rating color']
print("The rating color of the highest average rating is", color_of_avg_rating)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plot average rating by price range with highest rating highlighted
avg_rating_price_range = avg_rating_price_range.sort_values('Price range')
palette = ['yellow'] * len(avg_rating_price_range)
palette[np.argmax(avg_rating_price_range['Aggregate rating'])] = 'green'

plt.figure(figsize=(10, 6))
sns.barplot(x='Price range', y='Aggregate rating', data=avg_rating_price_range, palette=palette, errorbar=None)
plt.title('Average Rating by Price Range')
plt.xlabel('Price Range')
plt.ylabel('Average Rating')
plt.show()

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Calculate length of restaurant name and address
data['Length of restaurant name'] = data['Restaurant Name'].str.len()
data['Length of Address'] = data['Address'].str.len()

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Display first few rows of updated dataset
data.head()

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Display first few rows of specific columns for verification
data[['Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu']].head(10)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Encoding categorical columns
columns_to_convert = ['Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu']
for col in columns_to_convert:
    data[col] = data[col].str.lower()
    data[col] = data[col].map({'yes': 1, 'no': 0})

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Print updated data for verification
print(data[columns_to_convert].head())

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Display final dataset preview
data.head()
