"""FastAPI server exposing the trained GraphSAGE fraud detector.

Endpoints:
    GET  /api/health                  liveness probe
    GET  /api/options                 product/network/card-type dropdown values
    GET  /api/entities/{kind}         catalog of known device/card/address/email entities
    POST /api/predict                 score a transaction
"""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .inference import (
    Engine,
    TransactionRequest,
    ConsumerRequest,
    ENTITY_COLS,
    DEVICE_CATEGORY_LABEL,
    MERCHANT_CATEGORY_LABEL,
)
from .llm.client import initial_messages, stream_explanation
from .llm.retrieval import build_evidence


app = FastAPI(
    title="Sentinel GNN Fraud API",
    description="Graph Neural Network fraud detection over a Neo4j knowledge graph.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[Engine] = None


@app.on_event("startup")
def _load_engine() -> None:
    global engine
    engine = Engine.load()


class PredictPayload(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    product: str
    card_network: str
    card_type: str
    device: Optional[str] = None
    card: Optional[str] = None
    address: Optional[str] = None
    email_domain: Optional[str] = None


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": engine is not None}


@app.get("/api/options")
def options():
    if engine is None:
        raise HTTPException(503, "Model not loaded yet")
    return engine.options


@app.get("/api/entities/{kind}")
def entities(
    kind: str,
    limit: int = Query(500, ge=1, le=10_000),
    q: Optional[str] = None,
):
    if engine is None:
        raise HTTPException(503, "Model not loaded yet")
    if kind not in ENTITY_COLS:
        raise HTTPException(404, f"Unknown entity kind '{kind}'. Try one of {ENTITY_COLS}.")
    items = engine.entity_catalog[kind]
    if q:
        needle = q.lower()
        items = [r for r in items if needle in r['value'].lower()]
    return {"kind": kind, "total": len(items), "items": items[:limit]}


@app.post("/api/predict")
def predict(payload: PredictPayload):
    if engine is None:
        raise HTTPException(503, "Model not loaded yet")
    req = TransactionRequest(**payload.model_dump())
    return engine.predict(req)


# ── Consumer mode ───────────────────────────────────────────────────────


class ConsumerPayload(BaseModel):
    amount: float = Field(..., gt=0)
    email: Optional[str] = None
    device_category: str = 'other'
    merchant_category: str = 'shopping'
    card_network: str = 'visa'
    card_type: str = 'credit'


@app.get("/api/consumer/options")
def consumer_options():
    """Lay-friendly dropdown values for the consumer form."""
    if engine is None:
        raise HTTPException(503, "Model not loaded yet")
    return {
        'device_categories': [
            {'value': k, 'label': v} for k, v in DEVICE_CATEGORY_LABEL.items()
        ],
        'merchant_categories': [
            {'value': k, 'label': v} for k, v in MERCHANT_CATEGORY_LABEL.items()
        ],
        'card_network': engine.options['card_network'],
        'card_type': engine.options['card_type'],
    }


@app.post("/api/consumer/predict")
def consumer_predict(payload: ConsumerPayload):
    if engine is None:
        raise HTTPException(503, "Model not loaded yet")
    consumer_req = ConsumerRequest(**payload.model_dump())
    req, translation = consumer_req.to_transaction_request()
    result = engine.predict(req)
    result['mapping'] = translation     # show the user what we translated
    return result


# ── LLM Explainer (streaming chat) ──────────────────────────────────────


class ChatMessage(BaseModel):
    role: str               # 'user' | 'assistant'
    content: str


class ExplainPayload(BaseModel):
    """A flagged transaction the LLM should explain.

    `payload` is the request that produced `prediction` (analyst-mode shape
    after the consumer-mode mapping has been applied — so the LLM always
    reasons about the same set of fields). `messages` is the prior chat
    history for follow-up turns; empty on the first call (the server seeds
    the initial prompt).
    """
    payload: dict[str, Any]
    prediction: dict[str, Any]
    messages: list[ChatMessage] = Field(default_factory=list)


@app.post("/api/explain")
async def explain(body: ExplainPayload):
    if engine is None:
        raise HTTPException(503, "Model not loaded yet")

    evidence = build_evidence(engine, body.payload, body.prediction)

    # First turn: seed with the canonical "give me the headline" prompt.
    # Follow-up turns: trust whatever conversation the client sent.
    if body.messages:
        messages = [m.model_dump() for m in body.messages]
    else:
        messages = initial_messages()

    async def event_stream():
        async for event in stream_explanation(evidence, messages):
            # SSE wire format: "data: <json>\n\n" per event
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable proxy buffering for true streaming
        },
    )


@app.post("/api/explain/evidence")
def explain_evidence(body: ExplainPayload):
    """Return just the evidence packet — used by the UI for the
    'verify the citation' panel without round-tripping through Claude."""
    if engine is None:
        raise HTTPException(503, "Model not loaded yet")
    return build_evidence(engine, body.payload, body.prediction)
