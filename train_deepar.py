"""
DeepAR-Style LSTM Training Script
=================================
Custom PyTorch LSTM model matching DeepAR architecture.
Uses pure PyTorch to avoid framework compatibility issues.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# CONFIGURATION
# =============================================================================
CONTEXT_LEN = 64
PREDICTION_LEN = 10
NUM_FEATURES = 9
BATCH_SIZE = 64
MAX_EPOCHS = 50
HIDDEN_SIZE = 128
RNN_LAYERS = 2
LEARNING_RATE = 1e-3

DATA_PATH = "data/Location1.csv"
MODEL_PATH = "deepar_model.pt"

ALL_FEATURES = [
    'Power', 'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m', 
    'windspeed_10m', 'windspeed_100m', 'winddirection_10m', 
    'winddirection_100m', 'windgusts_10m'
]

print("="*70)
print("DeepAR-Style LSTM Training - Wind Power Forecasting")
print("="*70)
print(f"Context Length: {CONTEXT_LEN}")
print(f"Prediction Length: {PREDICTION_LEN}")
print(f"Architecture: {RNN_LAYERS} LSTM layers × {HIDDEN_SIZE} hidden units")
print("="*70)

# =============================================================================
# MODEL DEFINITION (DeepAR-style architecture)
# =============================================================================
class DeepARLSTM(nn.Module):
    """
    DeepAR-style LSTM for time series forecasting.
    Architecture matches Amazon DeepAR:
    - Stacked LSTM layers
    - Linear decoder for predictions
    """
    def __init__(self, input_size=9, hidden_size=128, num_layers=2, 
                 output_size=10, dropout=0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Stacked LSTM (DeepAR core)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection (DeepAR uses this for distribution params)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state for prediction
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        output = self.fc(last_hidden)     # (batch, output_size)
        
        return output
    
    @property
    def rnn(self):
        """Expose LSTM for direct access in attack scripts."""
        return self.lstm

# =============================================================================
# DATASET
# =============================================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, data, context_len, prediction_len):
        self.data = data
        self.context_len = context_len
        self.prediction_len = prediction_len
        self.n_samples = len(data) - context_len - prediction_len + 1
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Input: context window of all features
        x = self.data[idx:idx + self.context_len]
        # Target: next prediction_len values of Power (first column)
        y = self.data[idx + self.context_len:idx + self.context_len + self.prediction_len, 0]
        return torch.FloatTensor(x), torch.FloatTensor(y)

# =============================================================================
# LOAD AND PREPARE DATA (All locations)
# =============================================================================
print("\n[1/4] Loading data from all locations...")

import glob
DATA_DIR = "data"
csv_files = sorted(glob.glob(f"{DATA_DIR}/Location*.csv"))

all_dfs = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df = df.sort_values('Time').reset_index(drop=True)
    location_name = csv_file.split('/')[-1].replace('.csv', '')
    print(f"  Loaded {location_name}: {len(df)} timesteps")
    all_dfs.append(df)

# Concatenate all locations
combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"Total: {len(combined_df)} timesteps from {len(csv_files)} locations")

# Extract features and normalize
data = combined_df[ALL_FEATURES].values.astype(np.float32)
mean = data.mean(axis=0)
std = data.std(axis=0) + 1e-5
data = (data - mean) / std

# Save normalization params for later
norm_params = {'mean': mean.tolist(), 'std': std.tolist()}

print(f"Features: {ALL_FEATURES}")

# Split: 80% train, 20% validation
train_cutoff = int(len(data) * 0.8)
train_data = data[:train_cutoff]
val_data = data[train_cutoff:]

train_dataset = TimeSeriesDataset(train_data, CONTEXT_LEN, PREDICTION_LEN)
val_dataset = TimeSeriesDataset(val_data, CONTEXT_LEN, PREDICTION_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n[2/4] Creating datasets...")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# =============================================================================
# CREATE AND TRAIN MODEL
# =============================================================================
print("\n[3/4] Training DeepAR-style LSTM...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = DeepARLSTM(
    input_size=NUM_FEATURES,
    hidden_size=HIDDEN_SIZE,
    num_layers=RNN_LAYERS,
    output_size=PREDICTION_LEN,
    dropout=0.1
).to(device)

print(f"\nModel architecture:")
print(f"  Cell type: LSTM")
print(f"  Hidden size: {HIDDEN_SIZE}")
print(f"  RNN layers: {RNN_LAYERS}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

best_val_loss = float('inf')
patience_counter = 0
max_patience = 10

for epoch in range(MAX_EPOCHS):
    # Training
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1:3d}/{MAX_EPOCHS}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save({
            'model_state_dict': model.state_dict(),
            'context_len': CONTEXT_LEN,
            'prediction_len': PREDICTION_LEN,
            'hidden_size': HIDDEN_SIZE,
            'rnn_layers': RNN_LAYERS,
            'num_features': NUM_FEATURES,
            'feature_cols': ALL_FEATURES,
            'norm_params': norm_params,
        }, MODEL_PATH)
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# =============================================================================
# VERIFY
# =============================================================================
print("\n[4/4] Verifying saved model...")

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"  ✓ State dict keys: {len(checkpoint['model_state_dict'])}")
print(f"  ✓ Context length: {checkpoint['context_len']}")
print(f"  ✓ Prediction length: {checkpoint['prediction_len']}")

# Quick inference test
with torch.no_grad():
    x, y = next(iter(val_loader))
    x = x.to(device)
    pred = model(x)
    print(f"  ✓ Input shape: {x.shape}")
    print(f"  ✓ Output shape: {pred.shape}")

print("\n" + "="*70)
print("Training complete! Model ready for adversarial attacks.")
print(f"Model saved to: {MODEL_PATH}")
print("="*70)
