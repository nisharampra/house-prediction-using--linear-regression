# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (replace this with your own dataset)
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)

# View dataset columns
print(data.columns)

# Define feature variables (X) and the target variable (y)
X = data.drop(columns=["medv"])  # 'medv' is the target variable (Median Value of Homes)
y = data["medv"]

# Split the dataset into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Example prediction with proper feature names
example_data = pd.DataFrame([X_test.iloc[0].values], columns=X.columns)  # Use column names for prediction
predicted_price = model.predict(example_data)
print(f"Predicted price: {predicted_price[0]}")
