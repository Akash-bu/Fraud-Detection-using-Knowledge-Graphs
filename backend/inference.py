"""Inference layer for the trained GraphSAGE fraud detector.

A request describes a single (potentially-new) transaction. We append it as a
new node to the cached graph, wire bidirectional edges to whichever existing
device/card/address/email-domain nodes the request names, and run a forward
pass — so the GNN actually uses the historical context of those entities,
not just the request's own features.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Allow running from project root or from inside backend/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from build_graph import build_graph, ENTITY_COLS, SCALE_COLS  # noqa: E402
from graph_sage import GNN  # noqa: E402

DECISION_THRESHOLD = 0.4
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'model.pt'))
DF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'df.pkl'))


@dataclass
class TransactionRequest:
    amount: float
    product: str                 # 'C' | 'H' | 'R' | 'S'
    card_network: str            # 'visa' | 'mastercard' | ...
    card_type: str               # 'credit' | 'debit' | ...
    device: Optional[str] = None
    card: Optional[str] = None         # cards are integers in the data, accept as str
    address: Optional[str] = None      # addresses are floats in the data, accept as str
    email_domain: Optional[str] = None


# ── Consumer-mode mapping ──────────────────────────────────────────────
#
# The IEEE-CIS dataset is anonymized — `card`/`address` IDs aren't things a
# real person would know. Consumer mode lets a layman describe their
# transaction in plain language, and we map it to the closest equivalent
# the model was actually trained on.

MERCHANT_CATEGORY_TO_PRODUCT = {
    'shopping':     'C',   # online retail (Amazon, etc.)
    'subscription': 'R',   # recurring (Netflix, gym)
    'software':     'S',   # software / digital download
    'hosting':      'H',   # web hosting / cloud services
}

# Map lay device categories to the representative device fingerprint actually
# present in the training data, with its known fraud rate as displayed context.
DEVICE_CATEGORY_TO_FINGERPRINT = {
    'iphone':  'iOS Device',
    'android': 'SM-J700M Build/MMB29K',   # most common Android in the data
    'windows': 'Windows',
    'mac':     'MacOS',
    'other':   None,                       # unseen — no graph link
}

# Friendly labels for the UI (display only — not used by the model)
DEVICE_CATEGORY_LABEL = {
    'iphone':  'iPhone',
    'android': 'Android phone',
    'windows': 'Windows PC',
    'mac':     'Mac',
    'other':   'Other / unsure',
}

MERCHANT_CATEGORY_LABEL = {
    'shopping':     'Online shopping (e.g. Amazon)',
    'subscription': 'Subscription (e.g. Netflix, gym)',
    'software':     'Software / digital download',
    'hosting':      'Web hosting / cloud services',
}


@dataclass
class ConsumerRequest:
    """What a non-technical user can plausibly provide about their transaction."""
    amount: float
    email: Optional[str] = None
    device_category: str = 'other'         # iphone | android | windows | mac | other
    merchant_category: str = 'shopping'    # shopping | subscription | software | hosting
    card_network: str = 'visa'
    card_type: str = 'credit'

    def to_transaction_request(self) -> tuple["TransactionRequest", dict]:
        """Translate plain-language inputs into the model's vocabulary.

        Returns the technical request plus a dict describing how each lay
        input was translated, so the UI can be transparent about the mapping.
        """
        email_domain = None
        if self.email and '@' in self.email:
            email_domain = self.email.split('@', 1)[1].strip().lower()

        device = DEVICE_CATEGORY_TO_FINGERPRINT.get(self.device_category)
        product = MERCHANT_CATEGORY_TO_PRODUCT.get(self.merchant_category, 'C')

        req = TransactionRequest(
            amount=self.amount,
            product=product,
            card_network=self.card_network,
            card_type=self.card_type,
            device=device,
            card=None,             # consumers don't know their internal card ID
            address=None,          # consumers don't know their internal address code
            email_domain=email_domain,
        )
        translation = {
            'email_to_domain': {'input': self.email, 'mapped': email_domain},
            'device_to_fingerprint': {
                'input': DEVICE_CATEGORY_LABEL.get(self.device_category, self.device_category),
                'mapped': device,
            },
            'merchant_to_product': {
                'input': MERCHANT_CATEGORY_LABEL.get(self.merchant_category, self.merchant_category),
                'mapped': product,
            },
            'card_unmapped': True,
            'address_unmapped': True,
        }
        return req, translation


@dataclass
class Engine:
    """Loaded once at server startup. Holds the model + everything needed
    to turn a TransactionRequest into a feature vector and edges."""
    model: GNN
    context: dict
    df: pd.DataFrame
    median_timestamp: float
    options: dict = field(default_factory=dict)
    entity_catalog: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Engine":
        data, num_feat, context = build_graph(df_path=DF_PATH, save=False, return_context=True)

        model = GNN(in_channels=num_feat, hidden_channels=64, out_channels=2)
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location='cpu'))
        model.eval()

        df = pd.read_pickle(DF_PATH)
        median_ts = float(df['timestamp'].median())

        engine = cls(model=model, context=context, df=df, median_timestamp=median_ts)
        engine.options = engine._build_options()
        engine.entity_catalog = engine._build_entity_catalog()
        return engine

    # ── Catalog helpers ─────────────────────────────────────────────────

    def _build_options(self) -> dict:
        """Categorical option lists derived from the feature columns we trained on."""
        cols = self.context['feature_cols']
        def strip(prefix):
            return sorted(c[len(prefix):] for c in cols if c.startswith(prefix))
        return {
            'product':      strip('prod_'),
            'card_network': strip('net_'),
            'card_type':    strip('ctype_'),
        }

    def _build_entity_catalog(self) -> dict:
        """For each entity type, return a list of {value, txn_count, fraud_rate}
        suitable for populating the frontend dropdowns."""
        out = {}
        for col in ENTITY_COLS:
            stats = self.context['entity_stats'][col]
            items = []
            for value, s in stats.items():
                items.append({
                    'value': '' if pd.isna(value) else str(value),
                    'txn_count': int(s[f'{col}_txn_count']),
                    'fraud_rate': float(s[f'{col}_fraud_rate']),
                })
            # Show high-volume + interesting (fraudy) entities first
            items.sort(key=lambda r: (-r['fraud_rate'], -r['txn_count']))
            out[col] = items
        return out

    # ── Feature engineering for a single request ────────────────────────

    def _entity_lookup(self, col: str, value: Optional[str]):
        """Return (txn_count, fraud_rate, was_known)."""
        if value is None or value == '':
            return 0.0, 0.0, False
        stats = self.context['entity_stats'][col]
        # Stats are keyed by raw dataframe values (mix of str/int/float).
        # Try both the original string and a numeric cast.
        candidates = [value]
        for cast in (int, float):
            try:
                candidates.append(cast(value))
            except (ValueError, TypeError):
                pass
        for k in candidates:
            if k in stats:
                s = stats[k]
                return float(s[f'{col}_txn_count']), float(s[f'{col}_fraud_rate']), True
        return 0.0, 0.0, False

    def _entity_node_index(self, col: str, value: Optional[str]) -> Optional[int]:
        """Existing graph node id for the named entity, or None if unseen."""
        if value is None or value == '':
            return None
        le = self.context['encoders'][col]
        classes = set(le.classes_)
        for k in (value, str(value)):
            if k in classes:
                enc = int(le.transform([k])[0])
                return self.context['entity_offsets'][col] + enc
        return None

    def _build_feature_row(self, req: TransactionRequest, lookups: dict) -> np.ndarray:
        """Construct the 23-feature vector exactly as build_graph does, in the
        same column order. Operates on a single-row dict then aligns to
        feature_cols."""
        feature_cols = self.context['feature_cols']
        scaler = self.context['scaler']

        raw_scale = {
            'amount': req.amount,
            'timestamp': self.median_timestamp,
            'device_txn_count':       lookups['device'][0],
            'device_fraud_rate':      lookups['device'][1],
            'card_txn_count':         lookups['card'][0],
            'card_fraud_rate':        lookups['card'][1],
            'address_txn_count':      lookups['address'][0],
            'address_fraud_rate':     lookups['address'][1],
            'email_domain_txn_count': lookups['email_domain'][0],
            'email_domain_fraud_rate':lookups['email_domain'][1],
        }
        scale_vec = np.array([raw_scale[c] for c in SCALE_COLS], dtype=float).reshape(1, -1)
        scaled = scaler.transform(scale_vec).ravel()
        norm_lookup = dict(zip([f'{c}_norm' for c in SCALE_COLS], scaled))

        row = []
        for col in feature_cols:
            if col in norm_lookup:
                row.append(norm_lookup[col])
            elif col.startswith('prod_'):
                row.append(1.0 if col == f'prod_{req.product}' else 0.0)
            elif col.startswith('net_'):
                row.append(1.0 if col == f'net_{req.card_network}' else 0.0)
            elif col.startswith('ctype_'):
                row.append(1.0 if col == f'ctype_{req.card_type}' else 0.0)
            else:
                row.append(0.0)
        return np.array(row, dtype=np.float32)

    # ── Prediction ──────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, req: TransactionRequest) -> dict:
        lookups = {col: self._entity_lookup(col, getattr(req, col)) for col in ENTITY_COLS}
        feature_row = self._build_feature_row(req, lookups)

        base: torch.Tensor = self.context['data'].x
        base_edges: torch.Tensor = self.context['data'].edge_index
        new_idx = base.size(0)

        # Append new node
        new_x = torch.cat([base, torch.tensor(feature_row).unsqueeze(0)], dim=0)

        # Bidirectional edges from new node to each named entity that exists
        linked = {}
        new_src, new_dst = [], []
        for col in ENTITY_COLS:
            node_id = self._entity_node_index(col, getattr(req, col))
            linked[col] = node_id is not None
            if node_id is not None:
                new_src.extend([new_idx, node_id])
                new_dst.extend([node_id, new_idx])

        if new_src:
            extra = torch.tensor([new_src, new_dst], dtype=base_edges.dtype)
            new_edges = torch.cat([base_edges, extra], dim=1)
        else:
            new_edges = base_edges  # isolated node — relies only on its own features

        logits = self.model(new_x, new_edges)
        probs = F.softmax(logits[new_idx], dim=0)
        fraud_prob = float(probs[1].item())

        verdict = 'FRAUD' if fraud_prob >= DECISION_THRESHOLD else 'LEGIT'

        return {
            'fraud_probability': fraud_prob,
            'verdict': verdict,
            'threshold': DECISION_THRESHOLD,
            'explanation': self._explain(req, lookups, linked, fraud_prob),
        }

    def _explain(self, req: TransactionRequest, lookups: dict, linked: dict, fraud_prob: float) -> dict:
        """Surface the signals that most likely drove the prediction so the
        result reads like a fraud-analyst report, not just a black-box score."""
        signals = []
        for col in ENTITY_COLS:
            txn_count, rate, known = lookups[col]
            value = getattr(req, col)
            if not known:
                signals.append({
                    'label': col.replace('_', ' ').title(),
                    'value': value or '—',
                    'fraud_rate': None,
                    'txn_count': 0,
                    'note': 'Unseen — no historical context',
                    'severity': 'neutral',
                })
                continue
            severity = (
                'high' if rate >= 0.30 else
                'medium' if rate >= 0.10 else
                'low'
            )
            signals.append({
                'label': col.replace('_', ' ').title(),
                'value': value,
                'fraud_rate': rate,
                'txn_count': int(txn_count),
                'note': self._severity_note(rate, txn_count),
                'severity': severity,
            })

        # Amount sanity check vs train distribution
        amt = req.amount
        amt_mean = float(self.df['amount'].mean())
        amt_p99 = float(self.df['amount'].quantile(0.99))
        amount_note = None
        if amt > amt_p99:
            amount_note = f'Above 99th percentile (typical max ≈ ${amt_p99:,.0f})'
        elif amt < 5:
            amount_note = 'Very small — typical of card-testing transactions'

        return {
            'signals': signals,
            'amount': {
                'value': amt,
                'note': amount_note,
                'population_mean': amt_mean,
                'population_p99': amt_p99,
            },
            'summary': self._summary(fraud_prob, signals),
        }

    @staticmethod
    def _severity_note(rate: float, txn_count: int) -> str:
        pct = rate * 100
        if rate >= 0.30:
            return f'High historical fraud rate ({pct:.1f}% across {txn_count} txns)'
        if rate >= 0.10:
            return f'Elevated fraud rate ({pct:.1f}% across {txn_count} txns)'
        if txn_count >= 50:
            return f'Established entity — {pct:.2f}% fraud across {txn_count} txns'
        return f'Low volume ({txn_count} txns), {pct:.2f}% fraud'

    @staticmethod
    def _summary(fraud_prob: float, signals: list) -> str:
        high = [s for s in signals if s['severity'] == 'high']
        if fraud_prob >= 0.7 and high:
            top = ', '.join(s['label'] for s in high)
            return f'Strong fraud signal driven by historical patterns on {top}.'
        if fraud_prob >= 0.4:
            return 'Elevated risk — recommend manual review.'
        if fraud_prob >= 0.15:
            return 'Borderline — likely legitimate but worth monitoring.'
        return 'No significant fraud signal detected.'
