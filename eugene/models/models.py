import torch.nn as nn
import torch.nn.functional as F
import torch


# The original EUGENE architecture based on encoding prior knowledge into the model
# Idea is for first conv layer to learn GATA and ETS PWMs then to maxpool that information
# The second conv then learns combinations of these PWMs. Global max pooling
class otxCNN(nn.Module):
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
    def __init__(self, seqlen=66, first_channels=16, second_channels=8, first_kernel=16, second_kernel=5, maxpool_kernel=2, dropout=0.5):
        super(DeepSea, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=first_channels, kernel_size=first_kernel)
        self.Conv2 = nn.Conv1d(in_channels=first_channels, out_channels=second_channels, kernel_size=second_kernel)
        self.Batchnorm1 = nn.BatchNorm1d(first_channels)
        self.Batchnorm2 = nn.BatchNorm1d(second_channels)
        self.Maxpool = nn.MaxPool1d(kernel_size=maxpool_kernel, stride=1)
        self.Drop1 = nn.Dropout(p=dropout)
        self.linear_size = (seqlen - (first_kernel-1) - (second_kernel-1) - (maxpool_kernel-1))*second_channels
        self.Linear1 = nn.Linear(self.linear_size, 1)

        
    def forward(self, input):
        #print(input.size())
        x = self.Conv1(input)
        x = self.Batchnorm1(x)
        x = F.relu(x)
        #print(x.size())
        x = self.Conv2(x)
        x = self.Batchnorm2(x)
        x = F.relu(x)
        #print(x.size())
        x = self.Maxpool(x)
        #print(x.size())
        x = x.flatten(1)
        #print(x.size())
        x = self.Drop1(x)
        x = self.Linear1(x)
        return x


# A simple lstm architecture for single stranded dna input
# with adjustable hidden layers and nodes per layer
class ssLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=1, num_classes=1, dropout=0.3, bidirectional=False):
        super(ssLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
        else:
            self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Forward propagate LSTM
        out, _ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# A LSTM model that learns a representation of the forward sequence and it's reverse complement and then
# concatenates these representations and learns a linear layer to perform classification
class dsLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=1, num_classes=1, dropout=0.3, bidirectional=False):
        super(dsLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.reverse_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_size*4, num_classes)  # 4 for double stranded and bidirectional
        else:
            self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for double stranded
                        
    def forward(self, x, x_rev):     
        # Forward propagate LSTM
        out, _ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out_reverse, _ = self.reverse_lstm(x_rev)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = torch.cat((out, out_reverse), dim=2)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out
    
# A LSTM model that learns a representation of the forward sequence and it's reverse complement and then
# concatenates these representations and learns a linear layer to perform classification
class test_dsLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=1, num_classes=1, dropout=0.1, bidirectional=False):
        super(test_dsLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_size*8, num_classes)  # 4 for double stranded and bidirectional
        else:
            self.fc = nn.Linear(hidden_size*4, num_classes)  # 2 for double stranded
        self.dropout = nn.Dropout(dropout)
                        
    def forward(self, x, x_rev):     
        # Forward propagate LSTM
        out, _ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out_reverse, _ = self.lstm(x_rev)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = torch.cat((out, out_reverse), dim=2)
        #print(out.size())
        avg_pool = torch.mean(out, 1)
        max_pool, _ = torch.max(out, 1)
        #print(avg_pool.size(), max_pool.size())
        out = torch.cat((avg_pool, max_pool), 1)
        #print(out.size())
        out = self.fc(out)
        #out = self.fc(out[:, -1, :])
        out = self.dropout(out)


        return out