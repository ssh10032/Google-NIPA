import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb

class skeleton_LSTM(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(skeleton_LSTM, self).__init__()

        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.lstm1 = nn.LSTM(input_size=self.feature_dim, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=256, hidden_size=512, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, (hn, cn) = self.lstm3(x)
        x = F.relu(self.fc1(x[:, -1, :]))
        embedding = self.fc2(x)

        return embedding

class skeletonLSTM(nn.Module):
    def __init__(self, input_size, output_dim):
        super(skeletonLSTM, self).__init__()

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=1, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(128)

        self.lstm2 = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(256)

        self.lstm3 = nn.LSTM(input_size=256, hidden_size=512, num_layers=1, batch_first=True)
        self.layer_norm3 = nn.LayerNorm(512)

        self.lstm4 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)
        self.layer_norm4 = nn.LayerNorm(512)

        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        # LSTM layers with Layer Normalization
        x, _ = self.lstm1(x)
        x = self.layer_norm1(x)

        x, _ = self.lstm2(x)
        x = self.layer_norm2(x)

        x, _ = self.lstm3(x)
        x = self.layer_norm3(x)

        x, (hn, cn) = self.lstm4(x)
        x = self.layer_norm4(x)

        # Pooling to summarize sequence information
        x = torch.mean(x, dim=1)  # Mean pooling over the sequence

        # Fully connected layers with ReLU and Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        embedding = self.fc2(x)

        return embedding

class head(nn.Module):
    def __init__(self):
        super(head, self).__init__()

        # Feedforward layers
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)  # Output layer has 1 unit for binary classification
        self.sigmoid = nn.Sigmoid()  # Sigmoid for probability output

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class skeletonLSTM_bc(nn.Module):
    def __init__(self, input_size, output_dim):
        super(skeletonLSTM_bc, self).__init__()

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=1, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(128)

        self.lstm2 = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(256)

        self.lstm3 = nn.LSTM(input_size=256, hidden_size=512, num_layers=1, batch_first=True)
        self.layer_norm3 = nn.LayerNorm(512)

        self.lstm4 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)
        self.layer_norm4 = nn.LayerNorm(512)

        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, output_dim)
        self.classifier = nn.Linear(output_dim, 1)  # Binary classifier

    def forward(self, x):
        # LSTM layers with Layer Normalization
        x, _ = self.lstm1(x)
        x = self.layer_norm1(x)

        x, _ = self.lstm2(x)
        x = self.layer_norm2(x)

        x, _ = self.lstm3(x)
        x = self.layer_norm3(x)

        x, (hn, cn) = self.lstm4(x)
        x = self.layer_norm4(x)

        # Pooling to summarize sequence information
        x = torch.mean(x, dim=1)  # Mean pooling over the sequence

        # Fully connected layers with ReLU and Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        embedding = self.fc2(x)

        # Classification output
        classification_output = torch.sigmoid(self.classifier(embedding))

        return embedding, classification_output

