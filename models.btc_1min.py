import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from data_loader import crypto_data  # Assuming data_loader provides the data


# Data preparation function for PyTorch
def prepare_rnn_data(data, sequence_length=60):
    features = data[['open', 'high', 'low', 'close', 'vol_as_u']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    X, y = [], []
    for i in range(sequence_length, len(scaled_features)):
        X.append(scaled_features[i - sequence_length:i])
        y.append(scaled_features[i, 3])  # Predicting the 'close' price

    X, y = np.array(X), np.array(y)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    return X, y, scaler


# Prepare data
X, y, scaler = prepare_rnn_data(crypto_data)


# Define the RNN model
class PricePredictionRNN(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2, output_size=1):
        super(PricePredictionRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM layer
        out = out[:, -1, :]  # Take the last output in the sequence
        out = self.fc(out)  # Fully connected layer for output
        return out


# Define the SVR-style loss function with dynamic epsilon
class SVRLoss(nn.Module):
    def __init__(self, epsilon_factor=0.0001):  # 0.01% as 0.0001
        super(SVRLoss, self).__init__()
        self.epsilon_factor = epsilon_factor

    def forward(self, predictions, targets):
        # Calculate epsilon as 0.01% of the target prices
        epsilon = targets * self.epsilon_factor

        # Calculate absolute difference
        diff = torch.abs(predictions - targets)

        # Apply epsilon-insensitive loss
        loss = torch.where(diff > epsilon, diff - epsilon, torch.zeros_like(diff))
        return loss.mean()


# Instantiate the model
model = PricePredictionRNN(input_size=X.shape[2])

# Define custom SVR-style loss and optimizer
criterion = SVRLoss(epsilon_factor=0.0001)
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
    loss = criterion(outputs, y_train.unsqueeze(1))  # Match dimensions by unsqueezing y_train
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Switch to evaluation mode
model.eval()
with torch.no_grad():
    predicted = model(X_test).detach().numpy()
    predicted_prices = scaler.inverse_transform(np.concatenate((np.zeros((predicted.shape[0], 4)), predicted), axis=1))[
                       :, 4]

    # Actual prices for comparison
    actual_prices = scaler.inverse_transform(
        np.concatenate((np.zeros((y_test.shape[0], 4)), y_test.view(-1, 1).numpy()), axis=1))[:, 4]

# Plot the results
plt.plot(actual_prices, color='black', label='Actual Price')
plt.plot(predicted_prices, color='blue', label='Predicted Price')
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()