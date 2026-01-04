import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (CONTEXT_LEN, PREDICTION_LEN, NUM_FEATURES, BATCH_SIZE,
                    ACT_HIDDEN_SIZE, LEARNING_RATE, MAX_PONDERS, TIME_PENALTY,
                    DATA_PATH, ACT_MODEL_PATH, ALL_FEATURES)
from models.act import ACTModel
from utils.dataset import TimeSeriesDataset
from utils.data_loader import get_normalization_params

if __name__ == "__main__":
    print("="*70)
    print("Training ACT-LSTM (Adaptive Computation Time)")
    print("="*70)

    print("Loading data...")
    df = pd.read_csv(DATA_PATH).sort_values('Time')
    data = df[ALL_FEATURES].values.astype(np.float32)
    mean, std = get_normalization_params(data)
    data_norm = (data - mean) / std

    train_len = int(len(data_norm) * 0.8)
    train_ds = TimeSeriesDataset(data_norm[:train_len], CONTEXT_LEN, PREDICTION_LEN)
    val_ds = TimeSeriesDataset(data_norm[train_len:], CONTEXT_LEN, PREDICTION_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = ACTModel(NUM_FEATURES, ACT_HIDDEN_SIZE, PREDICTION_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    max_epochs = 30

    print("Starting training...", flush=True)

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        train_ponder = 0

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred, ponder = model(x)
            mse = criterion(pred, y)
            loss = mse + (TIME_PENALTY * ponder)
            loss.backward()
            optimizer.step()
            train_loss += mse.item()
            train_ponder += ponder.item()

        train_loss /= len(train_loader)
        train_ponder /= len(train_loader)

        model.eval()
        val_loss = 0
        val_ponder = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred, ponder = model(x)
                val_loss += criterion(pred, y).item()
                val_ponder += ponder.item()

        val_loss /= len(val_loader)
        val_ponder /= len(val_loader)

        print(f"Epoch {epoch+1:3d}: Loss={val_loss:.4f} | Avg Ponder Steps={val_ponder:.2f}/{MAX_PONDERS}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ACT_MODEL_PATH)

    print("\nTraining complete.")
    print(f"Model saved to {ACT_MODEL_PATH}")
