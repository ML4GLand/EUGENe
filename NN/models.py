import torch.nn as nn
import torch.nn.functional as F
import torch


# The original EUGENE architecture based on encoding prior knowledge into the model
# Idea is for first conv layer to learn GATA and ETS PWMs then to maxpool that information
# The second conv then learns combinations of these PWMs. Global max pooling
class EUGENE(nn.Module):
    def __init__(self):
        super(EUGENE, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=2, kernel_size=8)
        self.Conv2 = nn.Conv1d(in_channels=2, out_channels=3, kernel_size=4)
        self.Batchnorm1 = nn.BatchNorm1d(2)
        self.Batchnorm2 = nn.BatchNorm1d(3)
        self.Maxpool = nn.MaxPool1d(kernel_size=8, stride=1)
        self.Drop1 = nn.Dropout(p=0.5)
        self.Linear1 = nn.Linear(3, 1)

        
    def forward(self, input):
        # First conv block
        x = self.Conv1(input)
        x = self.Batchnorm1(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        
        # Second conv block with global max pooling
        x = self.Conv2(x)
        x = self.Batchnorm2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=x.size()[2:])

        # Fully connected block
        x = x.flatten(1)
        x = self.Drop1(x)
        x = self.Linear1(x)
        return x
    

# Model architecture adapted from DeepSea
class DeepSea(nn.Module):
    def __init__(self):
        super(DeepSea, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=16)
        self.Conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5)
        self.Batchnorm = nn.BatchNorm1d(32)
        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.Drop1 = nn.Dropout(p=0.5)
        self.Linear1 = nn.Linear(1472, 1)

        
    def forward(self, input):
        x = self.Conv1(input)
        x = self.Batchnorm(x)
        x = F.relu(x)
        x = self.Conv2(x)
        x = self.Batchnorm(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = x.flatten(1)
        x = self.Drop1(x)
        x = self.Linear1(x)
        return x


# A simple lstm architecture with adjustable hidden layers and nodes per layer
class lstm(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=1, num_classes=1, dropout=0.5, bidirectional=False):
        super(lstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
        else:
            self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial states
        #h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        #c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# A biLSTM model that learns a representation of the forward sequence and it's reverse complement and then 
# classifies based on their combination
class EUGENE_biLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=1, num_classes=1, dropout=0.3):
        super(EUGENE_biLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.reverse_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
    
    def forward(self, x, x_rev):
        
        # Forward propagate LSTM
        out, _ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out_reverse, _ = self.reverse_lstm(x_rev)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = torch.cat((out, out_reverse), dim=2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out