import torch
import torch.nn as nn

class CfCCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CfCCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Trainable Parameters for the Liquid Differential Equation
        self.ff1 = nn.Linear(input_size + hidden_size, hidden_size)
        self.ff2 = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Time-Constant Parameters (Learnable Physics)
        self.time_a = nn.Linear(1, hidden_size) # Learns sensitivity to time
        self.time_b = nn.Linear(1, hidden_size)

    def forward(self, x, h, time_delta):
        # Concatenate Input + Hidden State
        combined = torch.cat((x, h), dim=1)
        
        # Calculate the "Liquid Gate" (Exponential Decay)
        # This decides: "Do I remember or forget based on how much time passed?"
        t_gate = torch.sigmoid(self.time_a(time_delta) * time_delta + self.time_b(time_delta))
        
        # The Closed-form Solution (Hasani 2022)
        ff1_out = torch.tanh(self.ff1(combined))
        ff2_out = torch.tanh(self.ff2(combined))
        
        # Update Hidden State
        new_h = ff1_out * t_gate + ff2_out * (1.0 - t_gate)
        return new_h

class LiquidMalwareDetector(nn.Module):
    def __init__(self, input_features=3, hidden_units=64):
        super(LiquidMalwareDetector, self).__init__()
        self.hidden_units = hidden_units
        self.cfc = CfCCell(input_features, hidden_units)
        self.classifier = nn.Linear(hidden_units, 2) # [Safe, Malware]

    def forward(self, x, times):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_units)
        
        # Process the sequence packet by packet
        for t in range(x.size(1)):
            input_t = x[:, t, :]
            time_t = times[:, t, :]
            h = self.cfc(input_t, h, time_t)
            
        return torch.softmax(self.classifier(h), dim=1)