# Federated Learning (FL) on "smoking" dataset
# Algorithms: FedAvg, FedProx, FedNova
# Rounds=40, Clients=10, Local Epochs=10, LR=0.001

import os
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


CSV_PATH = "your_new_file.csv"
LOG_DIR = "./runs_FL"

ROUNDS = 40
NUM_CLIENTS = 10
LOCAL_EPOCHS = 10
LEARNING_RATE = 0.001
K_FEATURES = 30   # cap; will auto-adjust if dataset has fewer features
DIRICHLET_ALPHA = 0.5
RANDOM_STATE = 42

os.makedirs(LOG_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


df = pd.read_csv('/content/smoking.csv')
df.columns = df.columns.str.strip()


# Target labeling: 'smoking' -> 0/1
if 'smoking' not in df.columns:
    raise ValueError("The dataset must contain a 'smoking' column as the target.")

def to_binary(val):
    s = str(val).strip().lower()
    if s in {"1", "yes", "y", "true", "t"}:
        return 1
    if s in {"0", "no", "n", "false", "f"}:
        return 0
    # fallback: try numeric
    try:
        return 1 if float(s) > 0 else 0
    except:
        # Unknown: treat anything not clearly 'no' as 1
        return 1

df['smoking'] = df['smoking'].apply(to_binary).astype(int)

# Basic cleaning
# - Drop constant columns
# - Replace inf with NaN and drop
df = df.loc[:, (df != df.iloc[0]).any()]
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Ensure there are features left
feature_cols = [c for c in df.columns if c != 'smoking']
if len(feature_cols) == 0:
    raise ValueError("No feature columns found after cleaning.")

# Feature selection + Scaling + SMOTE
X = df[feature_cols].select_dtypes(include=[np.number])  # numeric features only (safe for scaler)
# If some features were non-numeric and all dropped, guard:
if X.shape[1] == 0:
    raise ValueError("No numeric features available. Convert categorical features to numeric first.")

y = df['smoking'].values

# First scale, then SelectKBest
scaler1 = StandardScaler()
X_scaled = scaler1.fit_transform(X)

k = min(K_FEATURES, X_scaled.shape[1])
selector = SelectKBest(f_classif, k=k)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = X.columns[selector.get_support()]

scaler2 = StandardScaler()
X_final = scaler2.fit_transform(df[selected_features])

# SMOTE to balance classes
X_bal, y_bal = SMOTE(random_state=RANDOM_STATE).fit_resample(X_final, y)

# Non-IID client partition (Dirichlet)
def dirichlet_partition(X, y, n_clients, alpha=0.5):
    labels = np.unique(y)
    n_samples = len(y)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X_sh = X[indices]
    y_sh = y[indices]

    client_data = [[] for _ in range(n_clients)]
    # For each label, split by Dirichlet proportions
    for label in labels:
        idxs = np.where(y_sh == label)[0]
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet([alpha] * n_clients)
        # turn proportions into split points
        split_points = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        splits = np.split(idxs, split_points)
        for cid, sp in enumerate(splits):
            client_data[cid].extend(sp.tolist())

    # Guarantee each client has at least some samples (very rare empty case fix)
    # If any client empty, steal from the largest client
    for cid in range(n_clients):
        if len(client_data[cid]) == 0:
            largest = max(range(n_clients), key=lambda k: len(client_data[k]))
            if len(client_data[largest]) > 1:
                client_data[cid].append(client_data[largest].pop())

    client_splits = []
    for cid in range(n_clients):
        idxs = np.array(client_data[cid], dtype=int)
        client_splits.append((X_sh[idxs], y_sh[idxs]))
    return client_splits


# Model
class IntrusionDetectionNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)



# Aggregation algorithms
def state_dict_avg(state_dicts):
    # average tensor by tensor
    avg = {}
    for k in state_dicts[0].keys():
        avg[k] = sum([sd[k] for sd in state_dicts]) / len(state_dicts)
    return avg

def fed_avg(updates):
    return state_dict_avg(updates)

def fed_prox(updates):
    # In many simple demos, FedProx server-side aggregation = FedAvg
    # (Proximal term is applied client-side; here we keep it simple as taught.)
    return state_dict_avg(updates)

def fed_nova(updates, local_steps):
    total_steps = sum(local_steps)
    weighted = {}
    for k in updates[0].keys():
        acc = 0
        for sd, n in zip(updates, local_steps):
            acc += sd[k] * (n / total_steps)
        weighted[k] = acc
    return weighted

class SmartContract:
    def __init__(self, strategy="fedavg", min_clients=2):
        self.strategy = strategy
        self.min_clients = min_clients

    def verify_and_aggregate(self, updates, **kwargs):
        if len(updates) < self.min_clients:
            return None
        if self.strategy == "fedavg":
            return fed_avg(updates)
        elif self.strategy == "fedprox":
            return fed_prox(updates)
        elif self.strategy == "fednova":
            return fed_nova(updates, kwargs.get("local_steps", [1]*len(updates)))
        else:
            raise ValueError("Unsupported strategy")

# Training loop (FL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = X_bal.shape[1]
num_classes = 2

strategies = ["fedavg", "fedprox", "fednova"]

results = {}

for strategy in strategies:
    print(f"\n=== Strategy: {strategy} | Clients: {NUM_CLIENTS} | Rounds: {ROUNDS} | Local epochs: {LOCAL_EPOCHS} | LR: {LEARNING_RATE} ===")
    results[strategy] = {}

    # Partition clients
    client_splits = dirichlet_partition(X_bal, y_bal, NUM_CLIENTS, alpha=DIRICHLET_ALPHA)
    clients = [(cx, cy) for cx, cy in client_splits]

    # Global model
    global_model = IntrusionDetectionNet(input_size, num_classes).to(device)
    contract = SmartContract(strategy=strategy, min_clients=max(2, NUM_CLIENTS // 10))
    writer = SummaryWriter(log_dir=f"{LOG_DIR}/{strategy}_{NUM_CLIENTS}")

    acc_list = []

    for r in range(ROUNDS):
        updates, local_steps = [], []

        for client_x, client_y in clients:
            model = IntrusionDetectionNet(input_size, num_classes).to(device)
            model.load_state_dict(global_model.state_dict())

            opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            criterion = nn.CrossEntropyLoss()

            xt = torch.tensor(client_x, dtype=torch.float32, device=device)
            yt = torch.tensor(client_y, dtype=torch.long, device=device)

            # local training
            for _ in range(LOCAL_EPOCHS):
                opt.zero_grad()
                out = model(xt)
                loss = criterion(out, yt)
                loss.backward()
                opt.step()

            updates.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            local_steps.append(len(client_x) * LOCAL_EPOCHS)

        # Aggregate
        if strategy == "fednova":
            agg = contract.verify_and_aggregate(updates, local_steps=local_steps)
        else:
            agg = contract.verify_and_aggregate(updates)

        if agg is not None:
            global_model.load_state_dict(agg)

        # Evaluate on full (balanced) set
        global_model.eval()
        with torch.no_grad():
            xt_all = torch.tensor(X_bal, dtype=torch.float32, device=device)
            yt_all = torch.tensor(y_bal, dtype=torch.long, device=device)
            logits = global_model(xt_all)
            preds = logits.argmax(dim=1)
            acc = (preds == yt_all).float().mean().item()
            acc_list.append(acc)
            writer.add_scalar("Accuracy", acc, r)

        print(f"Round {r+1:02d}/{ROUNDS} - Accuracy: {acc:.4f}")

    writer.close()
    results[strategy][NUM_CLIENTS] = {"accuracy": acc_list, "final_acc": acc_list[-1]}
    print(f"Final accuracy for {strategy} with {NUM_CLIENTS} clients: {acc_list[-1]:.4f}")


# Summary
print("\n===== FINAL SUMMARY =====")
for strategy in strategies:
    final_acc = results[strategy][NUM_CLIENTS]["final_acc"]
    print(f"{strategy.upper():7s} -> Final Acc: {final_acc:.4f}")
