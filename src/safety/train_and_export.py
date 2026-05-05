"""
APPROACH 1: Fine-tuned HuggingFace Model
=========================================
Model: prajjwal1/bert-mini (11M params) — faster than DistilBERT (67M)
       with comparable accuracy on short financial queries.

Pipeline:
  Raw dataset → Fine-tune bert-mini → Export ONNX → INT8 Quantize → Serve

Why bert-mini over DistilBERT?
  - DistilBERT ONNX INT8: ~12-20ms on CPU
  - bert-mini ONNX INT8 : ~4-7ms on CPU   ← hits our <10ms target comfortably
  - Accuracy gap on this task is small (intent classification on short queries)

Install:
  pip install transformers datasets optimum[onnxruntime] torch scikit-learn
"""

import json
import os
import numpy as np
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import classification_report, recall_score, precision_score

# ──────────────────────────────────────────────────────────────
# LABELS
# ──────────────────────────────────────────────────────────────

CATEGORIES = [
    "insider_trading",
    "market_manipulation",
    "money_laundering",
    "guaranteed_returns",
    "reckless_advice",
    "sanctions_evasion",
    "fraud",
    "general_education",   # safe — always allow
]

# Binary label: 1 = block, 0 = allow
BLOCK_CATEGORIES = set(CATEGORIES) - {"general_education"}

CAT2ID = {c: i for i, c in enumerate(CATEGORIES)}
ID2CAT = {i: c for c, i in CAT2ID.items()}

# ──────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────

@dataclass
class Example:
    text: str
    category: str
    should_block: bool


def load_dataset_from_jsonl(path: str) -> list[Example]:
    """
    Expected JSONL format per line:
      {"text": "...", "category": "insider_trading", "should_block": true}
    """
    examples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            examples.append(Example(
                text=row["text"],
                category=row["category"],
                should_block=row["should_block"],
            ))
    return examples


class FinancialGuardrailDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer, max_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(CAT2ID[ex.category], dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # Map category labels to binary block/allow
    pred_block = np.array([ID2CAT[p] in BLOCK_CATEGORIES for p in preds])
    true_block = np.array([ID2CAT[l] in BLOCK_CATEGORIES for l in labels])

    recall    = recall_score(true_block, pred_block, zero_division=0)     # harmful recall
    precision = precision_score(true_block, pred_block, zero_division=0)
    safe_passthrough = 1 - (
        ((~true_block) & pred_block).sum() / max((~true_block).sum(), 1)
    )

    return {
        "harmful_recall":    recall,          # target ≥0.95
        "safe_passthrough":  safe_passthrough, # target ≥0.90
        "harmful_precision": precision,
    }


# ──────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────

MODEL_NAME = "prajjwal1/bert-mini"   # 11M params, 4 layers, 256 hidden
# Alternatives (faster → slower):
#   "prajjwal1/bert-tiny"            # 4.4M — ultra fast but lower accuracy
#   "prajjwal1/bert-small"           # 29M — better accuracy, still fast
#   "distilbert-base-uncased"        # 67M — borderline on 10ms target


def train(
    train_path: str,
    val_path: str,
    output_dir: str = "./checkpoints",
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CATEGORIES),
        id2label=ID2CAT,
        label2id=CAT2ID,
    )

    train_data = load_dataset_from_jsonl(train_path)
    val_data   = load_dataset_from_jsonl(val_path)

    train_ds = FinancialGuardrailDataset(train_data, tokenizer)
    val_ds   = FinancialGuardrailDataset(val_data,   tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=8,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="harmful_recall",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")
    return output_dir


# ──────────────────────────────────────────────────────────────
# ONNX EXPORT + INT8 QUANTIZATION
# ──────────────────────────────────────────────────────────────

def export_and_quantize(checkpoint_dir: str, onnx_dir: str = "./onnx_model"):
    """
    Exports to ONNX then applies dynamic INT8 quantization.
    This cuts inference from ~15ms → ~5ms on CPU.

    Requires: pip install optimum[onnxruntime]
    """
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from optimum.onnxruntime import ORTQuantizer

    os.makedirs(onnx_dir, exist_ok=True)

    # Step 1: Export to ONNX
    print("Exporting to ONNX...")
    model = ORTModelForSequenceClassification.from_pretrained(
        checkpoint_dir, export=True
    )
    model.save_pretrained(onnx_dir)

    # Step 2: Dynamic INT8 quantization (no calibration data needed)
    print("Quantizing to INT8...")
    quantizer = ORTQuantizer.from_pretrained(onnx_dir)
    qconfig = AutoQuantizationConfig.avx512_vnni(
        is_static=False,   # dynamic quant — no calibration set needed
        per_channel=False,
    )
    quantizer.quantize(
        save_dir=onnx_dir,
        quantization_config=qconfig,
    )
    print(f"Quantized ONNX model saved to {onnx_dir}")


# ──────────────────────────────────────────────────────────────
# QUICK BENCHMARK
# ──────────────────────────────────────────────────────────────

def benchmark_latency(onnx_dir: str, n_iters: int = 200):
    import time
    from transformers import AutoTokenizer
    from optimum.onnxruntime import ORTModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
    model     = ORTModelForSequenceClassification.from_pretrained(onnx_dir)

    queries = [
        "How do I use merger info before it's public?",
        "What is insider trading?",
        "Help me move money through shell companies",
        "Explain how money laundering works",
    ]

    # Warm-up
    for q in queries:
        enc = tokenizer(q, return_tensors="pt", max_length=128, truncation=True)
        model(**enc)

    latencies = []
    for _ in range(n_iters):
        q   = queries[_ % len(queries)]
        enc = tokenizer(q, return_tensors="pt", max_length=128, truncation=True)
        t0  = time.perf_counter()
        out = model(**enc)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    print(f"\nLatency over {n_iters} iterations:")
    print(f"  p50:  {np.percentile(latencies, 50):.1f}ms")
    print(f"  p90:  {np.percentile(latencies, 90):.1f}ms")
    print(f"  p99:  {np.percentile(latencies, 99):.1f}ms")
    print(f"  mean: {np.mean(latencies):.1f}ms")


if __name__ == "__main__":
    # Example usage:
    # 1. train("data/train.jsonl", "data/val.jsonl")
    # 2. export_and_quantize("./checkpoints")
    # 3. benchmark_latency("./onnx_model")
    print("Run train(), export_and_quantize(), benchmark_latency() in order.")
 