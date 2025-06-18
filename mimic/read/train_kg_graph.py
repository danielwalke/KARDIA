from math import ceil
import random

import torch
import numpy as np
import os
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
from torch_geometric.data import Data
from mimic.orm_create.mimiciv_v3_orm import Labels, Note
from sqlalchemy import create_engine, and_, func
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn import global_mean_pool
from torch import optim
from torch.nn import functional as F
from sklearn import metrics
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


DB_URI = "postgresql://postgres:password@localhost:5432/mimicIV_v3"
engine = create_engine(DB_URI)
Session = sessionmaker(bind=engine)
session = Session()
def create_graph(dir):

    graph = json.load(open(os.path.join("revised_kgs", dir, "graph.json"), "r"))
    node_emb = np.load(os.path.join("revised_kgs", dir, "node_emb.npy"))
    edge_emb = np.load(os.path.join("revised_kgs", dir, "edge_emb.npy"))
    entity_dict = {key: i for i, key in enumerate(graph["entities"])}
    edge_dict = {key: i for i, key in enumerate(graph["edges"])}
    edge_idx, edge_attr = [], []
    for relation in graph["relations"]:
        src, rel, trg = relation
        edge_idx.append((entity_dict[src], entity_dict[trg]))
        edge_attr.append(edge_emb[edge_dict[rel]])
    edge_idx = np.array(edge_idx)
    edge_attr = np.array(edge_attr)
    edge_idx = torch.from_numpy(edge_idx).type(torch.int64)
    edge_attr = torch.from_numpy(edge_attr).type(torch.float)
    nodes = torch.from_numpy(node_emb).type(torch.float)

    label = session.query(Labels.label).where(Labels.row_id == dir).one_or_none()
    data = Data(x=nodes, edge_index=edge_idx, edge_attr=edge_attr, y=int(label[0]))
    torch.save(data, os.path.join("revised_kgs", dir, "data.pt"))
    return data

## TODO Update new KGs from remote dfg server
kg_dirs = os.listdir("revised_kgs")
data_graphs = []
db_note_row_ids = session.query(Note.row_id).filter(and_(func.lower(Note.text).not_like("%sepsis%"), func.lower(Note.text).not_like("%septic%"), func.lower(Note.text).not_like("%shock%"))).all()
db_note_row_ids = list(map(lambda n: n[0], db_note_row_ids))

kg_dirs = list(map(int, kg_dirs))
kg_dirs.sort()
for dir in tqdm(kg_dirs):
    if int(dir) not in db_note_row_ids: continue
    dir = str(dir)
    if not os.path.exists(os.path.join("revised_kgs", dir, "data.pt")):
        data = create_graph(dir)
    else:
        data = torch.load(os.path.join("revised_kgs", dir, "data.pt"), weights_only=False)

    if data.x.shape[0] <= 1:
        continue
    data.edge_index = data.edge_index.transpose(-1, 0)
    if data.edge_index.ndim == 1:
        data.edge_index = torch.tensor([[], []], dtype=torch.long)
    if data.edge_attr.ndim == 1:
        data.edge_attr = torch.tensor([], dtype=torch.float)
    data.y = torch.tensor([data.y], dtype=torch.float)
    data.x = torch.concatenate([data.x, torch.mean(data.x, dim = 0).unsqueeze(0)])
    graph = json.load(open(os.path.join("revised_kgs", dir, "graph.json"), "r"))
    graph["entities"] = list(map(lambda e: e, graph["entities"])) ## prevent nothing and use def. tokenizer -> Lets just map multiple tokens back to one word via sum of the tokenizer
    data.entities = "\t".join(graph["entities"])
    data.entities_list = graph["entities"]
    data.readout_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    data.readout_mask[-1] = 1
    new_src = torch.arange(data.x.shape[0])
    new_trg = torch.ones_like(new_src)*(data.x.shape[0]-1)
    readout_edges = torch.stack([new_src, new_trg])
    data.edge_index = torch.cat([data.edge_index, readout_edges], dim = -1)
    data.edge_attr = torch.cat([data.edge_attr, torch.ones((readout_edges.shape[-1], 768), dtype = torch.float)], dim = 0)
    data_graphs.append(data)

print(len(data_graphs))
train_idx, test_idx = train_test_split(np.arange(len(data_graphs)), test_size=0.2, stratify=[data.y for data in data_graphs], random_state=42)
train_data = [data_graphs[idx] for idx in train_idx]
test_data = [data_graphs[idx] for idx in test_idx]

vectorizer = TfidfVectorizer()
vectorizer.fit(list(map(lambda d: d.entities, train_data)))
tokenizer = vectorizer.build_tokenizer()
for data_set in [train_data, test_data]:
    for data in data_set:
        tokens = tokenizer(data.entities)
        node_idx = []
        for token in tokens:
            for node_i, entity in enumerate(data.entities_list):
                if token.lower() in entity.lower():
                    node_idx.append(node_i)
                    break
        node_idx = torch.tensor(node_idx, dtype=torch.long)
        vectorized_entities = vectorizer.transform([data.entities])
        vectorized_entities = vectorized_entities.toarray()[0]
        vectorized_entities = torch.from_numpy(vectorized_entities).float()
        num_features = vectorized_entities.shape[-1]
        num_nodes = data.x.shape[0]
        features = torch.zeros(num_nodes, num_features)
        #node_idx = torch.arange(num_nodes-1) # -1 for the aggregation readout node -> TODO consider mean aggregation as init for readout node

        feature_idx = torch.nonzero(vectorized_entities)

        if feature_idx.shape[0] == 0:
            print(feature_idx)
            print(data.entities)
            continue ## TODO extend vocabulary with further KGs or with an external vocubalry obtained form a db or KG?
        try:
            features[node_idx, feature_idx] = vectorized_entities[feature_idx]
        except Exception as e:
            print(data.entities)
            print(feature_idx)
            print(vectorizer.get_feature_names_out()[feature_idx])
            raise e
        data.x = features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_all_seeds(seed):
    """Set seeds for reproducibility."""
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    return g

def seed_worker(worker_id):
    """Set seed for DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# --- Model Definition ---
class GNN(torch.nn.Module):
    def __init__(self, hidden_dim, dropout, heads=1):
       super(GNN, self).__init__()
       # self.lin_x = torch.nn.Linear(len(vectorizer.get_feature_names_out()), hidden_dim)
       # self.lin_edge_attr = torch.nn.Linear(768, hidden_dim)
       self.conv1 = GATConv(len(vectorizer.get_feature_names_out()), hidden_dim,  add_self_loops=False, heads=heads, concat=True) #edge_dim=hidden_dim,
       self.conv2 = GATConv(hidden_dim*heads, 1, add_self_loops=False, concat=False) #, dropout=dropout edge_dim=hidden_dim,
       self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
       x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
       # x = torch.relu(self.lin_x(x))
       # # edge_attr = torch.relu(self.lin_edge_attr(edge_attr))
       # x = self.dropout(x)
       # # edge_attr = self.dropout(edge_attr)
       x = self.dropout(x)
       x = self.conv1(x, edge_index)#, edge_attr
       x = F.relu(x)
       x = self.dropout(x)
       # Applying dropout to edge attributes can also be beneficial
       # edge_attr = self.dropout(edge_attr)
       x = self.conv2(x, edge_index) #, edge_attr
       return x[data.readout_mask]

# --- Evaluation Function ---
def evaluate(model, loader, device):
    """Evaluates the model and returns AUROC and AUPRC."""
    with torch.inference_mode():
       model.eval()
       pred_probas = []
       y_trues = []
       for batch_data in loader:
          batch_data = batch_data.to(device)
          logit = model(batch_data)
          pred_proba = torch.sigmoid(logit)
          pred_probas.extend(pred_proba.cpu().tolist())
          y_trues.extend(batch_data.y.cpu().tolist())

    y_true = np.array(y_trues)
    y_pred_proba = np.array(pred_probas)
    auroc = metrics.roc_auc_score(y_true, y_pred_proba)
    auprc = metrics.average_precision_score(y_true, y_pred_proba)
    return auroc, auprc

print(torch.nonzero(train_data[0].x)[:20])
# --- Hyperparameter Optimization with Cross-Validation ---

# 1. Define the search space for Hyperopt (with epochs now tunable)
space = {
    "hidden_dim": hp.choice('hidden_dim', [8, 16, 32]),
    "dropout": hp.uniform('dropout', 0.1, 0.4),
    "lr": hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
    "weight_decay": hp.loguniform('weight_decay', np.log(1e-6), np.log(1e-3)),
    "heads": hp.choice('heads', [1, 2, 4]),
    "batch_size": hp.quniform('batch_size', 16, 128, 16),
    "epochs": hp.quniform('epochs', 10, 200, 5) # Epochs are now tunable
}

# Fixed parameters
N_SPLITS = 3
MAX_EVALS = 50
SEED = 0

# Extract labels for stratification
y_train_full = [d.y.item() for d in train_data]


# 2. Define the objective function for Hyperopt
def objective(params):
    """Objective function for hyperopt to minimize (negative AUROC)."""
    # Ensure integer parameters from hp.choice and hp.quniform are integers
    params['hidden_dim'] = int(params['hidden_dim'])
    params['batch_size'] = int(params['batch_size'])
    params['heads'] = int(params['heads'])
    params['epochs'] = int(params['epochs'])

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_aurocs = []

    print("-" * 30)
    print(f"Testing params: {params}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(train_data)), y_train_full)):
        # Subsets for current fold
        train_subset = [train_data[i] for i in train_idx]
        val_subset = [train_data[i] for i in val_idx]

        # DataLoaders for the current fold
        train_loader = DataLoader(train_subset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=params['batch_size'], shuffle=False)

        # Model and optimizer setup
        set_all_seeds(SEED)
        model = GNN(
            hidden_dim=params['hidden_dim'],
            dropout=params['dropout'],
            heads=params['heads']
        ).to(device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )

        # Loss function with pos_weight for the current training fold
        num_neg = sum(1 for data in train_subset if data.y.item() == 0)
        num_pos = sum(1 for data in train_subset if data.y.item() == 1)
        pos_weight = ceil(num_neg / num_pos) if num_pos > 0 else 1.0
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

        # Training loop using the tunable number of epochs
        for epoch in range(params['epochs']):
            model.train()
            for batch_data in train_loader:
                optimizer.zero_grad()
                batch_data = batch_data.to(device)
                logit = model(batch_data)
                loss = loss_fn(logit.squeeze(), batch_data.y)
                loss.backward()
                optimizer.step()

        # Evaluation for the fold
        auroc, _ = evaluate(model, val_loader, device)
        fold_aurocs.append(auroc)

    mean_auroc = np.mean(fold_aurocs)
    print(f"Mean Validation AUROC: {mean_auroc:.4f}")
    print("-" * 30)

    # Hyperopt minimizes the objective, so we return the negative of our metric
    return {'loss': -mean_auroc, 'status': STATUS_OK, 'params': params}


# 3. Run the hyperparameter optimization
trials = Trials()
# best_param_indices = fmin(
#     fn=objective,
#     space=space,
#     algo=tpe.suggest,
#     max_evals=MAX_EVALS,
#     trials=trials,
#     rstate=np.random.default_rng(SEED)
# )

# Extract the best performing hyperparameters
best_hyperparams = {'batch_size': 128, 'dropout': 0.0, 'epochs': 600, 'heads': 2, 'hidden_dim': 1, 'lr': .01, 'weight_decay': 1e-4} # trials.best_trial['result']['params']
print("\n" + "="*50)
print("      Hyperparameter Tuning Complete      ")
print("="*50)
#print(f"Best validation AUROC: {-trials.best_trial['result']['loss']:.4f}")
print("Best hyperparameters found:")
print(best_hyperparams)
print("="*50 + "\n")

# --- Final Model Training and Evaluation ---
print("Training final model on full training data with best hyperparameters...")

# Use a new generator for final training
final_generator = set_all_seeds(SEED)

# DataLoaders for final training
final_train_loader = DataLoader(
    train_data,
    batch_size=best_hyperparams["batch_size"],
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=final_generator
)
# Test loader does not need to be shuffled
test_loader = DataLoader(
    test_data,
    batch_size=best_hyperparams["batch_size"],
    shuffle=False
)


# Instantiate the final model
final_model = GNN(
    hidden_dim=best_hyperparams["hidden_dim"],
    dropout=best_hyperparams["dropout"],
    heads=best_hyperparams["heads"]
).to(device)

# optimizer = torch.optim.SGD(final_model.parameters(), lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False)
optimizer = optim.Adam(
    final_model.parameters(),
    lr=best_hyperparams["lr"],
    weight_decay=best_hyperparams["weight_decay"]
)

# Recalculate pos_weight on the full training data
pos_weight = ceil(sum([data.y[0] == 0 for data in train_data]) / sum([data.y[0] == 1 for data in train_data]))
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

# Final training loop using the best number of epochs found
num_epochs_final = best_hyperparams['epochs']
for epoch in range(num_epochs_final):
    epoch_loss = 0
    final_model.train()
    for batch_data in final_train_loader:
       optimizer.zero_grad()
       batch_data = batch_data.to(device)
       logit = final_model(batch_data)
       loss = loss_fn(logit.squeeze(), batch_data.y)
       loss.backward()
       optimizer.step()
       epoch_loss += loss.item()
    avg_loss = epoch_loss / len(final_train_loader)
    auroc_test, auprc_test = evaluate(final_model, test_loader, device)
    auroc_train, auprc_train = evaluate(final_model, final_train_loader, device)

    print(f"Epoch: {epoch} Train AUROC: {auroc_train:.4f} | Test AUROC: {auroc_test:.4f}")
    # print(f"Train AUPRC: {auprc_train:.4f} | Test AUPRC: {auprc_test:.4f}")
#    print(f'Epoch {epoch+1}/{num_epochs_final} | Final Training Loss: {avg_loss:.4f}')

#torch.save(final_model.state_dict(), 'final_model.pt')


print("\n--- Final Evaluation ---")
