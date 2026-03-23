# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 2: Create Sample Dataset

# Simulating retail sales data
data = {
    'Date': pd.date_range(start='2023-01-01', periods=200, freq='D'),
    'Sales': np.random.randint(50, 200, size=200)
}

df = pd.DataFrame(data)

# Convert Date to index
df.set_index('Date', inplace=True)

print("Sample Data:\n", df.head())

# Step 3: Data Preprocessing

# Normalize data (0 to 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Function to create sequences
def create_dataset(dataset, time_step=10):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X, y = create_dataset(scaled_data, time_step)

# Reshape for LSTM [samples, time_steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Step 4: Train-Test Split

train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

# Step 5: Build LSTM Model

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

# Step 6: Train Model

model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

# Step 7: Prediction

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Convert back to original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 8: Evaluation

mae = mean_absolute_error(y_test_actual, test_predict)
rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))

print("\nModel Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)

# Step 9: Visualization

plt.figure(figsize=(10,5))

plt.plot(y_test_actual, label='Actual Sales')
plt.plot(test_predict, label='Predicted Sales')

plt.title("Demand Forecasting (LSTM)")
plt.xlabel("Time")
plt.ylabel("Sales")
plt.legend()

plt.show()

# Step 10: Future Prediction

last_days = scaled_data[-time_step:]
last_days = last_days.reshape(1, time_step, 1)

future_prediction = model.predict(last_days)
future_prediction = scaler.inverse_transform(future_prediction)

print("\nNext Day Predicted Demand:", future_prediction[0][0])
