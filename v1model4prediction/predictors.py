import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

class UnifiedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(UnifiedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.minute_dense = nn.Linear(hidden_size, 10 * 2)  # 10 steps, mean and log-variance
        self.hour_dense = nn.Linear(hidden_size, 10 * 2)  # 10 steps, mean and log-variance

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        minute_output = self.minute_dense(lstm_out[:, -1, :]).view(-1, 10, 2)
        hour_output = self.hour_dense(lstm_out[:, -1, :]).view(-1, 10, 2)
        return minute_output, hour_output


def custom_loss(minute_pred, hour_pred, minute_true, hour_true):
    minute_mean, minute_log_var = minute_pred[:, :, 0], minute_pred[:, :, 1]
    hour_mean, hour_log_var = hour_pred[:, :, 0], hour_pred[:, :, 1]

    minute_loss = 0
    hour_loss = 0
    for i in range(4):  # Assuming 4 features: open, high, low, close
        if minute_true.shape[0] == 0 or minute_mean.shape[0] == 0:
            continue
        minute_loss += 0.5 * torch.mean(
            torch.exp(-minute_log_var) * (minute_true[:, :, i] - minute_mean) ** 2 + minute_log_var)
        if hour_true.shape[0] == 0 or hour_mean.shape[0] == 0:
            continue
        hour_loss += 0.5 * torch.mean(torch.exp(-hour_log_var) * (hour_true[:, :, i] - hour_mean) ** 2 + hour_log_var)

    return minute_loss + hour_loss


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length=600):
        self.data = data
        self.seq_length = seq_length
        self.features = self.data[['open', 'high', 'low', 'close', 'vol_as_u']].values
        self.targets = self.data[['open', 'high', 'low', 'close']].values

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]

        # Prepare future predictions for minute and hour levels
        y_minute = self.targets[idx + self.seq_length:idx + self.seq_length + 10]
        y_hour = self.targets[idx + self.seq_length * 6:idx + self.seq_length * 6 + 10]

        return torch.tensor(x, dtype=torch.float32), (
        torch.tensor(y_minute, dtype=torch.float32), torch.tensor(y_hour, dtype=torch.float32))


def collate_fn(batch):
    # Extract sequences and labels
    sequences, labels = zip(*batch)
    minute_labels, hour_labels = zip(*labels)

    # Pad sequences
    sequences_padded = pad_sequence(sequences, batch_first=True)
    minute_labels_padded = pad_sequence(minute_labels, batch_first=True)
    hour_labels_padded = pad_sequence(hour_labels, batch_first=True)

    return sequences_padded, (minute_labels_padded, hour_labels_padded)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create DataLoader
csv_file = 'btc_future_only_10s_a1.csv'
seq_length = 600  # Length of the sequence
batch_size = 32
data = pd.read_csv(csv_file)

# Define number of entries in last 10 hours
entries_in_10_hours = 36000

# Split the data
# train_data = data[:-entries_in_10_hours]
# test_data = data[-entries_in_10_hours:]

train_data = data[:2*entries_in_10_hours]
test_data = data[2*entries_in_10_hours:3*entries_in_10_hours]

# Create DataLoaders
train_dataset = TimeSeriesDataset(train_data, seq_length)
test_dataset = TimeSeriesDataset(test_data, seq_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn)


input_size = 5  # Adjust based on your data
hidden_size = 50  # Adjust based on your model's complexity

model = UnifiedLSTMModel(input_size, hidden_size).to(device)
criterion = custom_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop with Progress Display and Time Tracking
start_time = time.time()
for data, labels in tqdm(train_loader, desc='Training Progress'):
    data, minute_labels, hour_labels = data.to(device), labels[0].to(device), labels[1].to(device)
    optimizer.zero_grad()
    minute_pred, hour_pred = model(data)
    loss = criterion(minute_pred, hour_pred, minute_labels, hour_labels)
    loss.backward()
    optimizer.step()

elapsed_time = time.time() - start_time
print(f"Training completed in: {elapsed_time:.2f} seconds")

# Save the model
torch.save(model.state_dict(), 'unified_lstm_model.pth')
print("Model saved successfully.")


# Test Function
def test_model(model, test_loader):
    model.eval()
    minute_preds = []
    hour_preds = []
    minute_labels = []
    hour_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, minute_label, hour_label = data.to(device), labels[0].to(device), labels[1].to(device)
            minute_pred, hour_pred = model(data)
            minute_preds.append(minute_pred[:, :, 0].cpu().numpy())
            hour_preds.append(hour_pred[:, :, 0].cpu().numpy())
            minute_labels.append(minute_label.cpu().numpy())
            hour_labels.append(hour_label.cpu().numpy())

    return np.concatenate(minute_preds), np.concatenate(hour_preds), np.concatenate(minute_labels), np.concatenate(
        hour_labels)


# Visualization Function for Probability Distributions
def plot_probability_distributions(minute_preds, hour_preds, minute_labels, hour_labels):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    # Calculate and plot minute-level distributions
    for i in range(len(minute_preds)):
        minute_mean = minute_preds[i]
        minute_std = np.exp(minute_preds[i][:, 1] / 2)
        minute_true = minute_labels[i]

        x = np.linspace(minute_mean - 3 * minute_std, minute_mean + 3 * minute_std, 100)
        p = (1 / (minute_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - minute_mean) / minute_std) ** 2)

        ax[0].plot(x, p, label=f'Minute Data Sample {i}')
        ax[0].axvline(minute_true, color='r', linestyle='--')

    ax[0].set_title('Minute-Level Probability Distributions')
    ax[0].legend()

    # Calculate and plot hour-level distributions
    for i in range(len(hour_preds)):
        hour_mean = hour_preds[i]
        hour_std = np.exp(hour_preds[i][:, 1] / 2)
        hour_true = hour_labels[i]

        x = np.linspace(hour_mean - 3 * hour_std, hour_mean + 3 * hour_std, 100)
        p = (1 / (hour_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - hour_mean) / hour_std) ** 2)

        ax[1].plot(x, p, label=f'Hour Data Sample {i}')
        ax[1].axvline(hour_true, color='r', linestyle='--')

    ax[1].set_title('Hour-Level Probability Distributions')
    ax[1].legend()

    plt.tight_layout()
    plt.show()


# Load and test the model
model.load_state_dict(torch.load('unified_lstm_model.pth'))
minute_preds, hour_preds, minute_labels, hour_labels = test_model(model, test_loader)
plot_probability_distributions(minute_preds, hour_preds, minute_labels, hour_labels)
