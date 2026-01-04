import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (CONTEXT_LEN, PREDICTION_LEN, NUM_FEATURES, BATCH_SIZE,
                    MAX_EPOCHS, HIDDEN_SIZE, RNN_LAYERS, LEARNING_RATE,
                    DEEPAR_MODEL_PATH, ALL_FEATURES)
from models.deepar import DeepARLSTM
from utils.dataset import TimeSeriesDataset
from utils.data_loader import load_all_locations, get_normalization_params

print("="*70)
print("DeepAR-Style LSTM Training - Wind Power Forecasting")
print("="*70)

print("\n[1/4] Loading data from all locations...")
combined_df = load_all_locations("data")
print(f"Total: {len(combined_df)} timesteps")

data = combined_df[ALL_FEATURES].values.astype(np.float32)
mean, std = get_normalization_params(data)
data = (data - mean) / std
norm_params = {'mean': mean.tolist(), 'std': std.tolist()}

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

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

best_val_loss = float('inf')
patience_counter = 0
max_patience = 10

for epoch in range(MAX_EPOCHS):
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

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'context_len': CONTEXT_LEN,
            'prediction_len': PREDICTION_LEN,
            'hidden_size': HIDDEN_SIZE,
            'rnn_layers': RNN_LAYERS,
            'num_features': NUM_FEATURES,
            'feature_cols': ALL_FEATURES,
            'norm_params': norm_params,
        }, DEEPAR_MODEL_PATH)
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print("\n[4/4] Verifying saved model...")
checkpoint = torch.load(DEEPAR_MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    x, y = next(iter(val_loader))
    x = x.to(device)
    pred = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {pred.shape}")

print("\n" + "="*70)
print("Training complete!")
print(f"Model saved to: {DEEPAR_MODEL_PATH}")
print("="*70)
