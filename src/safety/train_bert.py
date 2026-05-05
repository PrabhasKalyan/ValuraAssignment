"""
Fine-tune prajjwal1/bert-mini on the 15-label financial-guardrail dataset.

Block decision: predicted_label.endswith('_op') → BLOCK. Otherwise ALLOW.
This forces the encoder to learn the OPERATIONAL vs EDUCATIONAL axis per
topic, using whole-sentence self-attention — the thing the linear model
couldn't do because its handcrafted intent feature is just a regex count.

After training: export to ONNX, dynamic INT8 quantize (~4–7ms/inference on CPU).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)
from sklearn.metrics import classification_report, recall_score, precision_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_bert"
CKPT = ROOT / "bert_ckpt"
ONNX = ROOT / "bert_onnx"

MODEL_NAME = "prajjwal1/bert-mini"  # 11M params, 4 layers, 256 hidden

# Load label mapping built by build_dataset_bert.py
labels_meta = json.load(open(DATA / "labels.json"))
LABELS = labels_meta["labels"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
BLOCK_LABEL_IDS = {i for i, l in ID2LABEL.items() if l.endswith("_op")}
print(f"Labels ({len(LABELS)}): {LABELS}")


# ──────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────
class JsonlDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=128):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        enc = self.tok(
            r["text"], max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(r["label_id"], dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    pred_block = np.array([p in BLOCK_LABEL_IDS for p in y_pred])
    true_block = np.array([t in BLOCK_LABEL_IDS for t in y_true])
    recall = recall_score(true_block, pred_block, zero_division=0)
    precision = precision_score(true_block, pred_block, zero_division=0)
    safe_pass = 1 - (((~true_block) & pred_block).sum() / max((~true_block).sum(), 1))
    return {
        "harmful_recall": recall,
        "safe_passthrough": safe_pass,
        "harmful_precision": precision,
        "exact_label_acc": float((y_pred == y_true).mean()),
    }


# ──────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────
def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = JsonlDataset(DATA / "train.jsonl", tok)
    val_ds = JsonlDataset(DATA / "val.jsonl", tok)

    use_mps = torch.backends.mps.is_available()
    args = TrainingArguments(
        output_dir=str(CKPT),
        num_train_epochs=8,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="exact_label_acc",  # harmful_recall saturates at 1.0; use per-label acc as tiebreaker
        greater_is_better=True,
        logging_steps=25,
        report_to="none",
        save_total_limit=1,
        # MPS / float16 not safe on Apple silicon for everything, leave fp32
        use_mps_device=use_mps and not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tok, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    trainer.save_model(str(CKPT))
    tok.save_pretrained(str(CKPT))
    print(f"\nSaved checkpoint to {CKPT}")

    # Final val report
    print("\nFinal eval on val set:")
    eval_out = trainer.evaluate(val_ds)
    for k, v in eval_out.items():
        if isinstance(v, float): print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
