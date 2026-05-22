import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler


def build_graph(df_path='data/df.pkl', save=True, seed=42):
    """Load df from pickle and build a PyG Data object with train/val/test splits."""

    df = pd.read_pickle(df_path)

    entity_cols = ['device', 'card', 'address', 'email_domain']
    for col in entity_cols:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col].astype(str).replace('nan', 'Unknown'))

    df['txn_enc'] = range(len(df))

    # Split indices first so device stats and scaler are fit on train only
    n = len(df)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(0.2 * n)
    n_val  = int(0.1 * n)
    test_idx  = perm[:n_test]
    val_idx   = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]

    is_train = np.zeros(n, dtype=bool)
    is_train[train_idx] = True

    for col in entity_cols:
        stats = df[is_train].groupby(col).agg(
            **{f'{col}_txn_count': ('fraud', 'count'),
               f'{col}_fraud_rate': ('fraud', 'mean')}
        ).reset_index()
        df = df.merge(stats, on=col, how='left')
        df[[f'{col}_txn_count', f'{col}_fraud_rate']] = \
            df[[f'{col}_txn_count', f'{col}_fraud_rate']].fillna(0)

    df['card_type'] = df['card_type'].replace('chare card', 'credit').fillna('Unknown')
    df['card_network'] = df['card_network'].fillna('Unknown')

    prod_dummies = pd.get_dummies(
        df['product'].fillna('Unknown'), prefix='prod', dtype=float
    )
    card_network_dummies = pd.get_dummies(
        df['card_network'], prefix='net', dtype=float
    )
    card_type_dummies = pd.get_dummies(
        df['card_type'], prefix='ctype', dtype=float
    )
    df = pd.concat([df, prod_dummies, card_network_dummies, card_type_dummies], axis=1)
    prod_cols = (list(prod_dummies.columns) + list(card_network_dummies.columns) + list(card_type_dummies.columns))

    scale_cols = [
        'amount', 'timestamp',
        'device_txn_count',       'device_fraud_rate',
        'card_txn_count',         'card_fraud_rate',
        'address_txn_count',      'address_fraud_rate',
        'email_domain_txn_count', 'email_domain_fraud_rate'
    ]
    norm_cols = [f'{c}_norm' for c in scale_cols]

    # Fit scaler on train rows only, transform all
    scaler = StandardScaler()
    scaler.fit(df.iloc[train_idx][scale_cols].fillna(0))
    df[norm_cols] = scaler.transform(df[scale_cols].fillna(0))

    feature_cols = norm_cols + prod_cols
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    x = torch.tensor(df[feature_cols].values, dtype=torch.float)
    y = torch.tensor(df['fraud'].astype(int).values, dtype=torch.long)

    # ── Edges — all 4 entity types bidirectional ──
    num_txns = len(df)
    offset = num_txns
    all_src, all_dst = [], []

    for col in entity_cols:
        enc  = torch.tensor(df[f'{col}_enc'].values, dtype=torch.long)
        txn  = torch.arange(num_txns, dtype=torch.long)
        eidx = enc + offset
        all_src.extend([txn, eidx])
        all_dst.extend([eidx, txn])
        offset += df[f'{col}_enc'].nunique()

    edge_index = torch.stack([
        torch.cat(all_src),
        torch.cat(all_dst)
    ], dim=0)

    # ── Pad all entity nodes ──
    total_entities = offset - num_txns
    entity_x = torch.zeros((total_entities, x.size(1)), dtype=torch.float)
    x = torch.cat([x, entity_x], dim=0)

    entity_y = torch.full((total_entities,), -1, dtype=torch.long)
    y = torch.cat([y, entity_y], dim=0)

    data = Data(x=x, edge_index=edge_index, y=y)

    # ── Apply masks ──
    total = len(y)
    data.train_mask = torch.zeros(total, dtype=torch.bool)
    data.val_mask   = torch.zeros(total, dtype=torch.bool)
    data.test_mask  = torch.zeros(total, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx]     = True
    data.test_mask[test_idx]   = True

    if save:
        torch.save(data, 'data/graph.pt')
        print(f"Saved {data}")

    return data, len(feature_cols)


if __name__ == "__main__":
    data, num_feat = build_graph()
    print(data)
    print(f"Train: {data.train_mask.sum()} | Val: {data.val_mask.sum()} | Test: {data.test_mask.sum()}")
