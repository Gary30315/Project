# I. Reading and Analyzing the Data

import pandas as pd
import numpy as np

train = pd.read_csv("LSTM/dataset/Google_Stock_Price_Train.csv",
                    index_col="Date",parse_dates=True)
# train.head()
# train.tail()

# Plot Open Price Data
from matplotlib import pyplot as plt
plt.figure()
plt.plot(train["Open"])
plt.title('Google stock open price ')
plt.ylabel('Price (USD)')
plt.xlabel('Days')  
plt.legend(['Open'], loc='upper left')
plt.show()


# II. Data Pre-processing

# Check Missing Value
print("checking if there exists any missing values \n",  train.isna().sum())

# Use open price for prediction (index = 1)
# Convert the pen price data into numpy array (keras only takes numpy array)
training_set = train.iloc[:, 1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)

# Set Sliding Window
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60: i, 0])
    y_train.append(training_set_scaled[i, 0]) 

# Convert to np.array    
X_train, y_train = np.array(X_train), np.array(y_train)

# Convert to LSTM 3D array format 
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# III. LSTM Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()

# Add the 1st LSTM layer: units = 80 (number of neurons)
# return_sequences = True => add more LSTM layer after the current one
# input_shape = (time_steps, feature) = (60, 1) in this example
model.add(LSTM(units = 150, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.3))

##add 2nd lstm layer
model.add(LSTM(units = 150, return_sequences = True))
model.add(Dropout(0.3))

# add 3rd lstm layer
model.add(LSTM(units = 150, return_sequences = True))
model.add(Dropout(0.3))

# add 4th lstm layer, return_sequences = False (no more LSTM layer will be added)
model.add(LSTM(units = 150, return_sequences = False))
model.add(Dropout(0.3))

# add output layer. output dimension = 1 (predicting 1 price each time).
model.add(Dense(units = 1))

# Compile LSTM model
model.compile(optimizer = 'adam', loss = 'mse')

# Train model
model.fit(x = X_train, y = y_train, batch_size = 16, epochs = 100)

# IV. Model Prediction
# Import test data
dataset_test = pd.read_csv('LSTM/dataset/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1: 2].values

dataset_total = pd.concat((train['Open'], dataset_test['Open']), axis = 0)

input_data = dataset_total[len(dataset_total)-len(dataset_test)- 60: ].values

# Reshape inputs to have only 1 column
re_input_data = input_data.reshape(-1, 1)

# Scale the data into (0,1)
inputs = sc.transform(re_input_data)

# Create Test dataset
X_test = []
for i in range(60, len(inputs)): 
    X_test.append(inputs[i-60: i, 0])
    
# Make test data as 3D numpy array, adding num of feature = 1
X_test = np.array(X_test)
X_test = np.reshape(X_test, newshape = (X_test.shape[0], X_test.shape[1], 1))

# Prediction
predicted_stock_price = model.predict(X_test)

# The prediction is in the scaled values. Need to reverse the prediction
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Plot prediction result
plt.plot(real_stock_price, color = 'red', label = 'Real price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted price')
plt.title('Google price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()



