"""
Production inference for the financial guardrail.

Loads the INT8-quantized ONNX bert-mini and the reason composer.
Output:
  {
    "blocked": bool,
    "topic": str,                # one of 6 harm topics or 'general'
    "intent": "operational" | "educational" | "general",
    "label": str,                # raw 13-class label
    "confidence": float,
    "reason": str,               # natural-language explanation
    "latency_ms": float,
  }
"""

import json
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

ROOT = Path(__file__).resolve().parent
ONNX_DIR = ROOT / "bert_onnx"
DATA_DIR = ROOT / "data_bert"

# Load label metadata (saved by build_dataset_bert.py)
labels_meta = json.load(open(DATA_DIR / "labels.json"))
LABELS = labels_meta["labels"]
ID2LABEL = {int(k): v for k, v in labels_meta["id2label"].items()}

BLOCK_THRESHOLD = 0.40   # operational + conf ≥ this → block
ALLOW_THRESHOLD = 0.30   # educational/general + conf ≥ this → allow

# ──────────────────────────────────────────────────────────
# Reason composer — maps (topic, intent, confidence) → English
# ──────────────────────────────────────────────────────────
TOPIC_BLURB = {
    "insider_trading":      "trading on non-public material information",
    "market_manipulation":  "artificially influencing market prices (e.g. spoofing, pump-and-dump, wash trading)",
    "money_laundering":     "concealing the origin of funds (e.g. structuring, layering, shell companies)",
    "guaranteed_returns":   "promoting investment returns as guaranteed or risk-free (a hallmark of Ponzi-style scams)",
    "reckless_advice":      "high-risk financial advice (e.g. all-in concentration, borrowing to speculate)",
    "sanctions_evasion":    "circumventing financial sanctions or embargoes",
}

OP_TEMPLATE = (
    "Blocked: this query reads as an OPERATIONAL request — i.e. the user appears "
    "to be asking how to perform {topic_blurb}. The intent is action-oriented, "
    "not educational. Confidence: {conf:.2f}."
)
OP_LOW_CONF_TEMPLATE = (
    "Blocked (low confidence): the query has signals consistent with operational "
    "intent around {topic_blurb}, but the model is not confident enough to "
    "categorise it precisely. Confidence: {conf:.2f}. Failing safe."
)
EDU_TEMPLATE = (
    "Allowed: this is an EDUCATIONAL query about {topic_blurb}. Discussing how a "
    "harmful activity works in conceptual or historical terms is not itself "
    "harmful. Confidence: {conf:.2f}."
)
GENERAL_TEMPLATE = (
    "Allowed: general financial question, no harmful intent detected. "
    "Confidence: {conf:.2f}."
)


def compose_reason(label: str, conf: float) -> str:
    if label == "general":
        return GENERAL_TEMPLATE.format(conf=conf)
    topic, intent = label.rsplit("_", 1)
    blurb = TOPIC_BLURB.get(topic, topic.replace("_", " "))
    if intent == "op":
        if conf < BLOCK_THRESHOLD:
            return OP_LOW_CONF_TEMPLATE.format(topic_blurb=blurb, conf=conf)
        return OP_TEMPLATE.format(topic_blurb=blurb, conf=conf)
    return EDU_TEMPLATE.format(topic_blurb=blurb, conf=conf)


# ──────────────────────────────────────────────────────────
# Classifier
# ──────────────────────────────────────────────────────────
class FinancialGuardrail:
    def __init__(self, onnx_dir: Path = ONNX_DIR):
        self.tok = AutoTokenizer.from_pretrained(str(onnx_dir))
        self.model = ORTModelForSequenceClassification.from_pretrained(
            str(onnx_dir), file_name="model_quantized.onnx"
        )

    def classify(self, query: str) -> dict:
        enc = self.tok(query, max_length=128, padding="max_length",
                       truncation=True, return_tensors="np")
        out = self.model(**enc)
        logits = out.logits[0]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        pid = int(np.argmax(probs))
        conf = float(probs[pid])
        label = ID2LABEL[pid]

        if label.endswith("_op"):
            should_block = True
            category = label[:-3]
        elif label.endswith("_edu"):
            should_block = False
            category = label[:-4]
        else:  # 'general' — internal-only, not surfaced. Pick closest _edu topic.
            should_block = False
            edu_idx = [i for i, l in ID2LABEL.items() if l.endswith("_edu")]
            best = max(edu_idx, key=lambda i: probs[i])
            category = ID2LABEL[best][:-4]

        return {
            "query": query,
            "should_block": should_block,
            "category": category,
        }

