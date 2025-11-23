import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# --- CONFIGURATION ---
# We will try Layer 9 once you extract it
DATA_FILE = "hubert_features_layer9.pkl" 
BATCH_SIZE = 64
LEARNING_RATE = 0.005 # Slightly higher for linear models
EPOCHS = 50

print(f"--- Training Linear Probe on {DATA_FILE} ---")
if not os.path.exists(DATA_FILE):
    print("Layer 9 file not found yet. Please run the extraction script!")
    exit()
    
df = pd.read_pickle(DATA_FILE)

# Labels
label_encoder = LabelEncoder()
df['label_id'] = label_encoder.fit_transform(df['label'])
classes = label_encoder.classes_

# Split
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(df, groups=df['speaker_id']))
train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

# Normalize
scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(np.stack(train_df['features'].values)), dtype=torch.float32)
y_train = torch.tensor(train_df['label_id'].values, dtype=torch.long)
X_test = torch.tensor(scaler.transform(np.stack(test_df['features'].values)), dtype=torch.float32)
y_test = torch.tensor(test_df['label_id'].values, dtype=torch.long)

# --- THE LINEAR MODEL (Simpler is Better) ---
class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProbe, self).__init__()
        # Just one layer. No hidden layers. No ReLU.
        self.layer = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.layer(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LinearProbe(768, len(classes)).to(device)

# Stronger Regularization (Weight Decay) to stop overfitting
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

print("Starting Training...")
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    predicted = torch.argmax(model(X_test), dim=1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)

print(f"--------------------------------")
print(f"Layer 9 Linear Accuracy: {accuracy*100:.2f}%")
print(f"--------------------------------")