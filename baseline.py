"""Tabular baseline: gradient-boosted trees on the same features the GNN uses.

Reuses build_graph() so the features and the train/val/test split are
identical to the GNN — any difference in AUC-PR is attributable to the
model, not the data prep. If this matches the GNN, the graph adds nothing.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from build_graph import build_graph

data, num_feat = build_graph(save=False)

X = data.x.numpy()
y = data.y.numpy()
train_mask = data.train_mask.numpy()
test_mask = data.test_mask.numpy()

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

# Up-weight fraud rows to match the GNN's class weighting
n_fraud = (y_train == 1).sum()
n_legit = (y_train == 0).sum()
pos_weight = n_legit / max(n_fraud, 1)
sample_weight = np.where(y_train == 1, pos_weight, 1.0)

clf = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.1,
    random_state=42,
)
clf.fit(X_train, y_train, sample_weight=sample_weight)

y_score = clf.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, y_score)
ap = average_precision_score(y_test, y_score)

print("=" * 60)
print(f"XGBoost-style baseline — {num_feat} features, same split as GNN")
print("=" * 60)
print(f"ROC-AUC: {roc:.4f}")
print(f"AUC-PR:  {ap:.4f}")
