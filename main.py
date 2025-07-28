import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv('house_data.csv')

# Features and target
X = data[['SquareFeet', 'Bedrooms', 'Bathrooms']]
y = data['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Results
print("Predicted Prices:", predictions)
print("R² Score:", r2_score(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))
