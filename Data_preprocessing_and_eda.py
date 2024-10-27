
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data = pd.read_csv('Dataset .csv')
data.head(5)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data.shape

data.info()

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
no_of_columns=len(data.columns)
print("Number of columns:",no_of_columns)

no_of_rows=data.shape[0]
print("Number of rows:",no_of_rows)



----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data.columns.tolist()

data.isnull().sum()

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data['Cuisines']= data['Cuisines'].fillna('')
Column_of_nullvalues=data['Cuisines'].mode()

print("Most frequent values in 'Cuisines':")
for mode in Column_of_nullvalues:
    print(mode)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data.isnull().sum()

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data.dtypes

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # #Before datatype conversion


data['Has Table booking'].unique()


columns_to_convert = [ 'Has Table booking','Has Online delivery', 'Is delivering now', 'Switch to order menu']

for col in columns_to_convert:
    data[col] = data[col].str.lower()
    data[col] = data[col].map({'yes': 1, 'no': 0})

# Print updated data to verify
print(data[columns_to_convert].head())

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # After datatype conversion


data['Has Online delivery'].unique()


data.head(2)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(10,6))
sns.histplot(data['Aggregate rating'],kde=True,bins=50)
plt.xlabel('Aggregate ratings')
plt.ylabel('Frequency')
plt.title('Distribution of the Aggregate ratings')
plt.show()

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
rating_counts=data['Aggregate rating'].value_counts()
print(rating_counts)


plt.figure(figsize=(10, 6))
sns.countplot(x='Aggregate rating', data=data, order=rating_counts.index)
plt.title('Distribution of Aggregate Rating')
plt.xlabel('Aggregate Rating')
plt.ylabel('Count')
plt.show()


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Discriptive statistics for all columns

data.describe(include='all')

#Discriptive statistics only for numeric columns

discriptive_of_data = data.describe()

print(discriptive_of_data)

#Descriptive statistics for a specified column

data[['Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']].describe()


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data['Country Code'].unique()


#Distribution of the country code 

plt.figure(figsize=(10,6))
sns.countplot(x= 'Country Code', data=data)
plt.xlabel('Country Code')
plt.ylabel('Frequency')
plt.title('Distribution of the Country code')
plt.show()

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Violin plot for Country Code and Price range

plt.figure(figsize=(12, 8))
sns.violinplot(x='Country Code', y='Price range', data=data)
plt.title('Distribution of Price Range across Country Codes')
plt.xlabel('Country Code')
plt.ylabel('Price Range')
plt.xticks(rotation=45)
plt.show()


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data['City'].unique()


plt.figure(figsize=(12, 8))
sns.countplot(x='City', data=data, order=data['City'].value_counts().index[:20])
plt.xticks(rotation=45)
plt.title('Count of Restaurants by City (Top 20)')
plt.xlabel('City')
plt.ylabel('Count')
plt.show()


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(12,6))
sns.pointplot(x='Country Code',y='Price range',data= data ,color='Green')
plt.xlabel('Country Code')
plt.ylabel('Price range')
plt.title('Price range over Country code')
plt.show()

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

data['Cuisines'].value_counts().sum

# %%
#top cuisines 

top_cuisines= data['Cuisines'].value_counts().head(10)
print(top_cuisines)

# %%
cuisine_counts = data['Cuisines'].value_counts()

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

plt.figure(figsize=(12, 8))
sns.barplot(x=cuisine_counts.head(20).values, y=cuisine_counts.head(20).index, palette="viridis")
plt.title('Top 20 Cuisines by Count')
plt.xlabel('Count')
plt.ylabel('Cuisine')
plt.show()

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#top cities


top_cities=data['City'].value_counts()
print(top_cities)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

plt.figure(figsize=(12, 8))
sns.barplot(x=top_cities.head(10).values, y=top_cities.head(10).index, palette="viridis")
plt.title('Top 10 Cities with the Highest Number of Restaurants')
plt.xlabel('Number of Restaurants')
plt.ylabel('City')
plt.show()

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import folium

m = folium.Map(location=(data['Latitude'].mean(), data['Longitude'].mean()), zoom_start=12, tiles='CartoDB positron')

for index, row in data.iterrows():
    folium.CircleMarker([row['Latitude'], row['Longitude']], radius=3, color='blue', fill_color='blue').add_to(m)

m

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#distribution of the restaurant across different cities

plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='City', order=data['City'].value_counts().head(10).index)
plt.title('Distribution of Restaurants Across Cities')
plt.xlabel('City')
plt.ylabel('Number of Restaurants')
plt.grid(axis='y' , linestyle='--')
plt.show()

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder

#Label encoding to convert the city to numeric value

label_encoding = LabelEncoder()
data['City_Encoded'] = label_encoding.fit_transform(data['City'])

correlation = data['City_Encoded'].corr(data['Aggregate rating'])

print("Correlation between City and Rating:",correlation)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
correlation_dict = {
    'City_Encoded': {
        'Aggregate rating': correlation
    },
    'Aggregate rating': {
        'City_Encoded': correlation
    }
}

plt.figure(figsize=(8,6))
sns.heatmap(pd. DataFrame(correlation_dict), cmap="crest", annot=True, fmt=".2f", linewidths=0.5 )
plt.title('Correlation between location and rating')
plt.show()

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
corr_long_rating = data['Longitude'].corr(data['Aggregate rating'])
corr_latt_rating = data['Latitude'].corr(data['Aggregate rating'])

print("Correlation between Latitude and Aggregate rating:", corr_latt_rating)
print("Correlation between Longitude and Aggregate rating:", corr_long_rating)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Correlation of location and rating in scatterplot


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Latitude', y='Aggregate rating', data=data, label='Latitude vs Rating')
sns.scatterplot(x='Longitude', y='Aggregate rating', data=data, label='Longitude vs Rating')
plt.xlabel('Latitude / Longitude')
plt.ylabel('Aggregate Rating')
plt.title('Correlation between Location and Rating')
plt.legend()
plt.show()

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
correlation_matrix = data[['Latitude', 'Longitude', 'Aggregate rating']].corr()
correlation_matrix

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# # Correlation of location and rating using Heatmap


plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix , cmap="crest", annot=True, fmt=".2f", linewidths=0.5) 
plt.title("Correlation of location and rating")
plt.show()





