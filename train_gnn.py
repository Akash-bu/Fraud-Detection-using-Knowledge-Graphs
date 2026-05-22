import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
from build_graph import build_graph
from graph_sage import GNN

data, num_feat = build_graph()
print(data)

model = GNN(in_channels=num_feat, hidden_channels=64, out_channels=2)

weights = torch.tensor([1.0, 3.6], dtype=torch.float)  # sqrt(110059/8607) ≈ 3.6

crit = torch.nn.CrossEntropyLoss(weight=weights)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    opt.zero_grad()
    out = model(data.x, data.edge_index)
    loss = crit(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    opt.step()
    return loss.item()

def validate():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        y_score = F.softmax(out, dim=1)[data.val_mask, 1].numpy()
    y_true = data.y[data.val_mask].numpy()
    y_pred = (y_score > 0.5).astype(int)
    auc_pr = average_precision_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return auc_pr, f1

best_val = 0
for epoch in range(1, 201):
    loss = train()
    if epoch % 10 == 0:
        auc_pr, f1 = validate()
        marker = " **" if auc_pr > best_val else ""
        if auc_pr > best_val:
            best_val = auc_pr
            torch.save(model.state_dict(), 'data/model.pt')
        print(f"Epoch {epoch:3d} | loss = {loss:.4f} | val_AUC-PR = {auc_pr:.4f} | val_F1 = {f1:.4f}{marker}")

print(f"\nBest val AUC-PR: {best_val:.4f} — model saved.")
