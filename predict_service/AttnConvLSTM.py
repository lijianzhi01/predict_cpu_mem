import torch
import pandas as pd  
import numpy as np  
import torch.nn as nn  
from sklearn.preprocessing import MinMaxScaler  
from sklearn.metrics import r2_score  
import torch.nn.functional as F  
import torch  

class AttnConvLSTM(nn.Module):  
    def __init__(self, input_size, hidden_size, num_layers, output_size, kernel_size):  
        super(AttnConvLSTM, self).__init__()  
        self.hidden_size = hidden_size  
        self.num_layers = num_layers  
  
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size, padding=(kernel_size-1)//2)  
        self.fc = nn.Linear(hidden_size*2, output_size)  
  
    def forward(self, x):  
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  
  
        lstm_out, _ = self.lstm(x, (h_0, c_0))  
        conv_out = self.conv(x.transpose(1,2)).transpose(1,2)  
  
        # Attention mechanism  
        attn_weights = F.softmax(torch.bmm(lstm_out, conv_out.transpose(1,2)), dim=-1)  
        attn_applied = torch.bmm(attn_weights, conv_out)  
  
        out = torch.cat((lstm_out, attn_applied), 2)  
        out = self.fc(out[:, -1, :])  
  
        return out