import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, average_precision_score
)

from graph_sage import GNN
from build_graph import build_graph

data, num_feat = build_graph(save=False)
model = GNN(in_channels=num_feat, hidden_channels=64, out_channels=2)
model.load_state_dict(torch.load('data/model.pt', weights_only=True))
model.eval()

with torch.no_grad():
    out = model(data.x, data.edge_index)
    probs = F.softmax(out, dim = 1)
    pred = out.argmax(dim=1)

y_true = data.y[data.test_mask].numpy()
y_score = probs[data.test_mask, 1].numpy()

threshold = 0.4
y_pred = (y_score > threshold).astype(int)

print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_true, y_pred, target_names=['Legit', 'Fraud'], digits=4))

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
cm = confusion_matrix(y_true, y_pred)
print(f"              Pred Legit | Pred Fraud")
print(f"True Legit    {cm[0][0]:6d}     | {cm[0][1]:6d}")
print(f"True Fraud    {cm[1][0]:6d}     | {cm[1][1]:6d}")

print("=" * 60)
print("PROBABILITY METRICS")
print("=" * 60)
print(f"ROC-AUC: {roc_auc_score(y_true, y_score):.4f}")
print(f"AUC-PR:  {average_precision_score(y_true, y_score):.4f}")