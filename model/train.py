import torch
import torch.nn as nn
import torch.optim as optim
from src.preprocessing import TrafficPreprocessor
from src.model import LiquidMalwareDetector

def train():
    # 1. Prepare Data
    print("Loading Data...")
    processor = TrafficPreprocessor(seq_length=50)
    # Using dummy data for demonstration if csv missing
    try:
        X, T, y = processor.process_csv('data/traffic_log.csv')
    except:
        print("CSV not found. Generating dummy training data...")
        X = torch.randn(100, 50, 3) # 100 samples, 50 packets, 3 features
        T = torch.abs(torch.randn(100, 50, 1)) # Positive time deltas
        y = torch.randint(0, 2, (100,))

    # 2. Setup Model
    model = LiquidMalwareDetector(input_features=3, hidden_units=64)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Loop
    print("Starting Training...")
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X, T)
        loss = criterion(outputs, y.long())
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/10 | Loss: {loss.item():.4f}")

    # 4. Save Weights
    torch.save(model.state_dict(), "liquid_model.pth")
    print("Training Complete. Model Saved.")

if __name__ == "__main__":
    train()