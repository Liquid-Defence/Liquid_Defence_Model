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

    # 2. Split Data (80% Train, 20% Validation)
    dataset_size = len(X)
    train_size = int(0.8 * dataset_size)
    
    X_train, X_val = X[:train_size], X[train_size:]
    T_train, T_val = T[:train_size], T[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # 3. Setup Model
    model = LiquidMalwareDetector(input_features=3, hidden_units=64)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Loop
    print("Starting Training...")
    best_val_loss = float('inf')
    
    for epoch in range(10):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train, T_train)
        loss = criterion(outputs, y_train.long())
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val, T_val)
            val_loss = criterion(val_outputs, y_val.long())
            
            # Metric Calculation
            _, predicted = torch.max(val_outputs, 1)
            correct = (predicted == y_val).sum().item()
            accuracy = correct / len(y_val)
            
        print(f"Epoch {epoch+1}/10 | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {accuracy:.4f}")
        
        # Checkpointing
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), "liquid_model.pth")
            print("  --> Best Model Saved!")

    print("Training Complete.")

if __name__ == "__main__":
    train()