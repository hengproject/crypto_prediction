import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data_loader import crypto_data  # Assuming data_loader provides the data

# Prepare the data without scaling
def prepare_rnn_data(data, sequence_length=60):
    # Extract relevant columns
    features = data[['open', 'high', 'low', 'close', 'vol_as_u']].values

    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])  # Sequence of features
        y.append(features[i, 3])  # Predicting the 'close' price only

    X, y = np.array(X), np.array(y)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    return X, y

# Prepare data without scaling
X, y = prepare_rnn_data(crypto_data)

# Define the RNN model
class PricePredictionRNN(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2, output_size=1):
        super(PricePredictionRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)   # LSTM layer
        out = out[:, -1, :]     # Take the last output in the sequence
        out = self.fc(out)      # Fully connected layer for output
        return out

# Instantiate the model
model = PricePredictionRNN(input_size=X.shape[2])

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
    loss = criterion(outputs.squeeze(), y_train)  # Squeeze outputs to match y_train dimensions
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Switch to evaluation mode
model.eval()
with torch.no_grad():
    predicted = model(X_test).detach().cpu().numpy().flatten()  # Flatten to 1D array for easy plotting
    actual = y_test.cpu().numpy()  # No need for inverse scaling since we worked with raw prices

# Plot the results
plt.plot(actual, color='black', label='Actual Price')
plt.plot(predicted, color='blue', label='Predicted Price')
plt.title('Price Prediction without Scaling')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
