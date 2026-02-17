import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

class TrafficPreprocessor:
    def __init__(self, seq_length=50):
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()

    def process_csv(self, filepath):
        # Load Raw Data
        df = pd.read_csv(filepath)
        
        # 1. Sort by Time (Crucial for Liquid Networks)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
        
        # 2. Calculate Time-Delta (The Physics Input)
        # "How much time passed since the last packet?"
        df['time_delta'] = df['timestamp'].diff().dt.total_seconds().fillna(0.0)
        
        # 3. Select Features (Metadata Only - No Payload)
        features = df[['packet_size', 'protocol', 'direction']].values
        time_deltas = df[['time_delta']].values
        labels = df['is_malware'].values
        
        # 4. Normalize (Scale to 0-1)
        features = self.scaler.fit_transform(features)
        
        return self.create_sequences(features, time_deltas, labels)

    def create_sequences(self, features, times, labels):
        X, T, y = [], [], []
        # Create sliding windows of 50 packets
        for i in range(len(features) - self.seq_length):
            X.append(features[i : i + self.seq_length])
            T.append(times[i : i + self.seq_length])
            y.append(labels[i + self.seq_length])
            
        return (torch.FloatTensor(np.array(X)), 
                torch.FloatTensor(np.array(T)), 
                torch.FloatTensor(np.array(y)))