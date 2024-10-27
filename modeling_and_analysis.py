
# # Restaurant Ratings Analysis

# ## Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings

# Set display options
%matplotlib inline
warnings.filterwarnings('ignore')

------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ## Loading Data
data = pd.read_csv("Dataset .csv")
data.head(10)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ## Initial Data Analysis
# ### Display Columns
data.columns.tolist()

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ### Encoding Categorical Columns
columns_to_convert = ['Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu']
for col in columns_to_convert:
    data[col] = data[col].str.lower()
    data[col] = data[col].map({'yes': 1, 'no': 0})

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Print updated data to verify
print(data[columns_to_convert].head())

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ## Filtering Data
# Filter restaurants that offer both table booking and online delivery
filtered_data = data[(data['Has Table booking'] == 1) & (data['Has Online delivery'] == 1)]
filtered_data[['Has Table booking', 'Has Online delivery']].head(3)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ## Feature Selection
# Select features for modeling
X = data[['Average Cost for two', 'Price range', 'Has Table booking', 'Has Online delivery', 'Votes']]
y = data['Aggregate rating']

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ## Scaling Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ## Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ## Model Training
# ### Linear Regression
reg = LinearRegression()
fitting_lr = reg.fit(X_train, y_train)
y_pred_lr = reg.predict(X_test)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ### Model Evaluation
# Add constant for OLS regression
X_with_constant = sm.add_constant(X_scaled)
results = sm.OLS(y, X_with_constant).fit()
results.summary()

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Compare actual vs predicted values
comparison_df = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_lr})
comparison_df['Difference'] = comparison_df['Actual value'] - comparison_df['Predicted value']
print(comparison_df.head())

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Calculate metrics
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print('Linear Regression:')
print(f'Mean Absolute Error (MAE): {mae_lr:.2f}')
print(f'R-squared (R2): {r2_lr:.2f}')

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ### Decision Tree Regression
dt_reg = DecisionTreeRegressor()
fitting_dt = dt_reg.fit(X_train, y_train)
y_pred_dt = dt_reg.predict(X_test)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Evaluate Decision Tree
mae_dt = mean_absolute_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print('Decision Tree:')
print(f'Mean Absolute Error (MAE): {mae_dt:.2f}')
print(f'R-squared (R2): {r2_dt:.2f}')

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ### Random Forest Regression
rf_reg = RandomForestRegressor()
fitting_rf = rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Evaluate Random Forest
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print('Random Forest:')
print(f'Mean Absolute Error (MAE): {mae_rf:.2f}')
print(f'R-squared (R2): {r2_rf:.2f}')

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ## Model Comparison
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor()
}

metrics = {
    'Model': [],
    'Mean Absolute Error (MAE)': [],
    'R-squared (R2)': []
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics['Model'].append(model_name)
    metrics['Mean Absolute Error (MAE)'].append(mae)
    metrics['R-squared (R2)'].append(r2)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Create DataFrame
metrics_df = pd.DataFrame(metrics)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Display the DataFrame
print(metrics_df)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ## Data Visualization
# ### Cuisines and Ratings
data[['Cuisines', 'Aggregate rating']].head(5)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Get top 20 cuisines by count
top_cuisines = data['Cuisines'].value_counts().head(20).index
filtered_data = data[data['Cuisines'].isin(top_cuisines)]
cuisine_ratings = filtered_data.groupby('Cuisines')['Aggregate rating'].mean().reset_index()

plt.figure(figsize=(12, 8))
sns.barplot(x='Aggregate rating', y='Cuisines', data=cuisine_ratings, palette='viridis', alpha=0.595, edgecolor='blue')
plt.xlabel('Aggregate Ratings')
plt.ylabel('Cuisines')
plt.title('Top 20 Cuisines and Aggregate Ratings')
plt.show()

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ### Votes and Popular Cuisines
top_cuisines = data['Cuisines'].value_counts().head(20).index
filtered_data = data[data['Cuisines'].isin(top_cuisines)]
cuisine_votes_sum = filtered_data.groupby('Cuisines')['Votes'].sum()
most_pop_cuisines = cuisine_votes_sum.sort_values(ascending=False)

plt.figure(figsize=(12, 8))
most_pop_cuisines.head(10).plot(kind='bar', color='orange', alpha=0.524, edgecolor='red')
plt.xlabel('Cuisines')
plt.ylabel('Votes')
plt.title('Most Popular Cuisines Based on Votes')
plt.grid(axis='y', linestyle='--')
plt.show()

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ### Highest Rated Cuisines
cuisines_ratings = data.groupby('Cuisines')['Aggregate rating'].mean()
sort_cuisines_ratings = cuisines_ratings.sort_values(ascending=False)
top_cuisines_ratings = sort_cuisines_ratings.head(10)
top_cuisines_ratings_df = pd.DataFrame(top_cuisines_ratings)

plt.figure(figsize=(10, 6))
top_cuisines_ratings.head(10).plot(kind='bar', color='skyblue', alpha=0.596, edgecolor='blue')
plt.xlabel('Cuisines')
plt.ylabel('Aggregate Rating')
plt.title('Highest Aggregate Ratings Based on Cuisines')
plt.show()

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ### Distribution of Aggregate Ratings
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Aggregate rating', kde=True, color='skyblue', alpha=0.5, edgecolor='grey')
plt.title('Distribution of Ratings Using Histogram')
plt.xlabel('Aggregate Rating')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Aggregate rating', color='orange')
plt.title('Distribution of Ratings Using Boxplot')
plt.xlabel('Aggregate Rating')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=data, x='Aggregate rating', color='green')
plt.title('Distribution of Aggregate Ratings Using Violin Plot')
plt.xlabel('Aggregate Rating')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--')
plt.show()

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ### Ratings by Top Cuisines
top_cuisines = data['Cuisines'].value_counts().head(10).index
filtered_data = data[data['Cuisines'].isin(top_cuisines)]

plt.figure(figsize=(12, 8))
sns.boxplot(data=filtered_data, x='Aggregate rating', y='Cuisines', palette='pastel')
plt.xlabel('Aggregate Ratings')
plt.ylabel('Cuisines')
plt.title('Box Plot of Aggregate Ratings by Top 20 Cuisines')
plt.xticks(rotation=90)
plt.show()

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ### Ratings by Top Cities
top_10_cities = data['City'].value_counts().head(10).index
filtered_cities = data[data['City'].isin(top_10_cities)]

plt.figure(figsize=(12, 8))
sns.violinplot(data=filtered_cities, x='City', y='Aggregate rating', palette='cubehelix')
plt.xlabel('Cities')
plt.ylabel('Aggregate Ratings')
plt.title('Aggregate Ratings of Different Cities Using Violin Plot')
plt.xticks(rotation=90)
plt.show()

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ### Pair Plot of Selected Features
features = ['Average Cost for two', 'Price range', 'Has Table booking', 'Has Online delivery', 'Votes', 'Aggregate rating']
data_subset = data[features]

plt.figure(figsize=(12, 8))
sns.pairplot(data_subset, diag_kind='kde')
plt.suptitle('Pair Plot of Features and Aggregate Rating', y=1.02)
plt.show()
