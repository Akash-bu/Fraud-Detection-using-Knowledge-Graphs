"""Subgraph retrieval for the LLM explainer.

For each flagged transaction we assemble a structured **evidence dict** —
the facts the model is allowed to cite in its narrative. Pulled from the
in-memory dataframe (which is the materialized projection of the Neo4j
graph), so retrieval is fast and deterministic.

Each fact carries an ID (`device_fraud_rate`, `peer_txn_3105398`, …) that
the LLM must reference in its citations. The validator can then reject any
response that cites IDs not in the evidence dict — closing the hallucination
loop without paying for a second LLM call.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# Reuse the inference engine's entity column list so retrieval stays in sync
from ..inference import ENTITY_COLS, Engine


def build_evidence(
    engine: Engine,
    payload: dict,
    prediction: dict,
    *,
    max_peers_per_entity: int = 5,
) -> dict:
    """Return a structured evidence packet for the explainer.

    The shape is intentionally JSON-friendly — the LLM consumes it as a
    serialized block in the system prompt and cites facts by ID.
    """
    df = engine.df

    txn = {
        'amount': float(payload['amount']),
        'product': payload.get('product'),
        'card_network': payload.get('card_network'),
        'card_type': payload.get('card_type'),
        'verdict': prediction['verdict'],
        'fraud_probability': round(float(prediction['fraud_probability']), 4),
        'decision_threshold': prediction['threshold'],
    }

    entities = {}
    for col in ENTITY_COLS:
        value = payload.get(col)
        entities[col] = _entity_facts(engine, df, col, value)

    amount = _amount_context(df, float(payload['amount']))

    peers = {}
    for col in ENTITY_COLS:
        value = payload.get(col)
        if value:
            peers[col] = _peer_transactions(df, col, value, k=max_peers_per_entity)

    return {
        'transaction': txn,
        'entities': entities,
        'peers': peers,
        'amount_context': amount,
        'population': {
            'total_transactions': int(len(df)),
            'fraud_rate_overall': round(float(df['fraud'].mean()), 4),
        },
    }


def _entity_facts(engine: Engine, df: pd.DataFrame, col: str, value: Optional[str]) -> dict:
    """Per-entity facts the model can cite. Includes a 'known' flag so the
    LLM can correctly attribute unseen entities to lack of historical data
    rather than to clean history."""
    if value is None or value == '':
        return {'present': False, 'value': None, 'known': False}

    stats = engine.context['entity_stats'].get(col, {})
    candidates = [value]
    for cast in (int, float):
        try:
            candidates.append(cast(value))
        except (ValueError, TypeError):
            pass

    matched = None
    for k in candidates:
        if k in stats:
            matched = stats[k]
            break

    if matched is None:
        return {'present': True, 'value': str(value), 'known': False,
                'note': 'Entity not in training data — no historical context'}

    txn_count = int(matched[f'{col}_txn_count'])
    fraud_rate = float(matched[f'{col}_fraud_rate'])
    return {
        'present': True,
        'value': str(value),
        'known': True,
        'txn_count': txn_count,
        'fraud_rate': round(fraud_rate, 4),
        'fraud_count': int(round(fraud_rate * txn_count)),
        'severity': _severity(fraud_rate),
    }


def _amount_context(df: pd.DataFrame, amount: float) -> dict:
    """Where this amount sits in the training distribution."""
    s = df['amount']
    pct = float((s <= amount).mean())
    return {
        'value': round(amount, 2),
        'percentile': round(pct, 4),
        'mean': round(float(s.mean()), 2),
        'median': round(float(s.median()), 2),
        'p95': round(float(s.quantile(0.95)), 2),
        'p99': round(float(s.quantile(0.99)), 2),
        'max': round(float(s.max()), 2),
    }


def _peer_transactions(df: pd.DataFrame, col: str, value: str, k: int) -> list[dict]:
    """Up to k example transactions sharing the same entity value — useful
    for grounding statements like 'this device has 70 prior frauds'."""
    candidates = [value]
    for cast in (int, float):
        try:
            candidates.append(cast(value))
        except (ValueError, TypeError):
            pass

    mask = df[col].isin(candidates)
    sub = df[mask]
    if sub.empty:
        return []

    # Prefer fraud examples first (more interesting evidence), then high-amount
    sub = sub.sort_values(['fraud', 'amount'], ascending=[False, False])
    out = []
    for _, row in sub.head(k).iterrows():
        out.append({
            'txn_id': int(row['txn_id']),
            'amount': round(float(row['amount']), 2),
            'fraud': bool(row['fraud']),
            'product': str(row['product']),
        })
    return out


def _severity(fraud_rate: float) -> str:
    if fraud_rate >= 0.30:
        return 'high'
    if fraud_rate >= 0.10:
        return 'medium'
    return 'low'
