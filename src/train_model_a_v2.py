import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler # <--- NEW IMPORTS
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURATION ---
# Use the filename directly since you moved it to the folder
DATA_FILE = "mfcc_features.pkl"  
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50 # Increased epochs

print(f"--- Loading Data from {DATA_FILE} ---")
try:
    df = pd.read_pickle(DATA_FILE)
except FileNotFoundError:
    # Fallback to full path if you didn't move it
    df = pd.read_pickle("E:/Speech_Data_Cache/mfcc_features.pkl")

# 1. Prepare Labels
label_encoder = LabelEncoder()
df['label_id'] = label_encoder.fit_transform(df['label'])
num_classes = len(label_encoder.classes_)
print(f"Classes: {label_encoder.classes_}")

# 2. Split Data (Grouped by Speaker to avoid cheating)
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(df, groups=df['speaker_id']))

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

# 3. Feature Processing & NORMALIZATION (The Fix!)
# Convert lists to numpy arrays
X_train_raw = np.stack(train_df['features'].values)
X_test_raw = np.stack(test_df['features'].values)

# Initialize Scaler (Fits only on TRAIN data to prevent leaking info)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Convert to Tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train = torch.tensor(train_df['label_id'].values, dtype=torch.long)

X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test = torch.tensor(test_df['label_id'].values, dtype=torch.long)

print(f"Training shape: {X_train.shape}")

# 4. Improved Model Architecture (Added Dropout & more neurons)
class BetterAccentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BetterAccentClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) # Helps stabilize training
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) # Prevents overfitting
        
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.layer1(x))))
        x = self.dropout(self.relu(self.bn2(self.layer2(x))))
        x = self.layer3(x)
        return x

model = BetterAccentClassifier(input_dim=13, hidden_dim=128, output_dim=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. Training Loop
print("\n--- Starting Training V2 ---")
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

# 6. Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)

print("------------------------------------------------")
print(f"Model A (MFCC + Norm) Accuracy: {accuracy * 100:.2f}%")
print(f"Baseline to beat: {100/num_classes:.2f}%")
print("------------------------------------------------")