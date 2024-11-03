import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from data_loader import crypto_data
from sklearn.metrics import mean_absolute_error, r2_score


# Load and limit the data
crypto_data = crypto_data.head(1000)

# Prepare the data with MinMaxScaler
def prepare_rnn_data(data, sequence_length=60):
    features = data[['open', 'high', 'low', 'close', 'vol_as_u']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    X, y = [], []
    for i in range(sequence_length, len(scaled_features)):
        X.append(scaled_features[i-sequence_length:i].flatten())  # Flatten to use in a fully connected network
        y.append(scaled_features[i, 3])  # Predicting the normalized 'close' price

    X, y = np.array(X), np.array(y)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    return X, y, scaler

# Prepare data
X, y, scaler = prepare_rnn_data(crypto_data)

# Define the baseline model
class BaselineModel(nn.Module):
    def __init__(self, input_size):
        super(BaselineModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # First fully connected layer
        self.fc2 = nn.Linear(64, 32)          # Second fully connected layer
        self.fc3 = nn.Linear(32, 1)           # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model
input_size = X.shape[1]  # Flattened input size: sequence_length * number of features
model = BaselineModel(input_size=input_size)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Split data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Training the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs.squeeze(), y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Switch to evaluation mode
model.eval()
with torch.no_grad():
    predicted = model(X_test).detach().cpu().numpy().flatten()

    # Revert predictions to the original scale using inverse_transform
    placeholder = np.zeros((predicted.shape[0], 5))
    placeholder[:, 3] = predicted
    predicted_prices = scaler.inverse_transform(placeholder)[:, 3]

    placeholder_actual = np.zeros((y_test.shape[0], 5))
    placeholder_actual[:, 3] = y_test.cpu().numpy()
    actual_prices = scaler.inverse_transform(placeholder_actual)[:, 3]

# Plot the results
plt.plot(actual_prices, color='black', label='Actual Price')
plt.plot(predicted_prices, color='blue', label='Predicted Price')
plt.title('Baseline Model Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Calculate evaluation metrics
mae = mean_absolute_error(actual_prices, predicted_prices)
r2 = r2_score(actual_prices, predicted_prices)

# Print metrics
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}")