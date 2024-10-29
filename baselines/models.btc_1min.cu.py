import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from baselines.data_loader import crypto_data

# Prepare the data with prior data only for prediction
def prepare_rnn_data(data, sequence_length=60):
    features = data[['open', 'high', 'low', 'close', 'vol_as_u']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    X, y = [], []
    for i in range(sequence_length, len(scaled_features) - 1):  # Stop at len(scaled_features) - 1
        X.append(scaled_features[i-sequence_length:i])  # Past data up to time i
        y.append(scaled_features[i, 3])  # Target is the next close price at i + 1

    X, y = np.array(X), np.array(y)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    return X, y, scaler

# Prepare the data
X, y, scaler = prepare_rnn_data(crypto_data)

# Define the model
class PricePredictionRNN(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2, output_size=1):
        super(PricePredictionRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM layer
        out = out[:, -1, :]    # Take the last output in the sequence
        out = self.fc(out)     # Fully connected layer for output
        return out

# Instantiate and train the model as before
model = PricePredictionRNN(input_size=X.shape[2])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

    if (epoch+1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
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
import matplotlib.pyplot as plt
plt.plot(actual_prices, color='black', label='Actual Price')
plt.plot(predicted_prices, color='blue', label='Predicted Price')
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
