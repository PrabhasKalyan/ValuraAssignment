"""Export the fine-tuned bert-mini to ONNX, then dynamic INT8 quantize."""
import shutil
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "bert_ckpt"
ONNX_FP32 = ROOT / "bert_onnx_fp32"
ONNX = ROOT / "bert_onnx"  # final INT8 model

print("1/2 Exporting to ONNX (fp32)...")
m = ORTModelForSequenceClassification.from_pretrained(str(CKPT), export=True)
m.save_pretrained(str(ONNX_FP32))
# Carry the tokenizer over
for f in CKPT.iterdir():
    if f.name.startswith(("tokenizer", "vocab", "special_tokens")):
        shutil.copy(f, ONNX_FP32 / f.name)

print("2/2 Dynamic INT8 quantization...")
quantizer = ORTQuantizer.from_pretrained(str(ONNX_FP32))
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
quantizer.quantize(save_dir=str(ONNX), quantization_config=qconfig)

# Carry tokenizer + label config to the quantized dir too
for f in CKPT.iterdir():
    if f.name.startswith(("tokenizer", "vocab", "special_tokens", "config.json")):
        if not (ONNX / f.name).exists():
            shutil.copy(f, ONNX / f.name)

print(f"\nDone. Quantized ONNX model at: {ONNX}")
import os
fp32_size = sum(f.stat().st_size for f in ONNX_FP32.iterdir() if f.is_file())
int8_size = sum(f.stat().st_size for f in ONNX.iterdir() if f.is_file())
print(f"FP32 ONNX size:  {fp32_size / 1e6:.1f} MB")
print(f"INT8 ONNX size:  {int8_size / 1e6:.1f} MB")
