from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


MODELS = [
    ("ORIGINAL FP16", "kaist_ots_best.pt", True),
    ("ONNX", "kaist_ots.onnx", False),
    ("TensorRT FP16", "kaist_ots_fp16.engine", False),
    ("TensorRT INT8", "kaist_ots_int8_mixed.engine", False),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", default="results/reproduced")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    for label, filename, use_half in MODELS:
        model_path = models_dir / filename
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        print(f"\nEvaluating {label}: {model_path}")
        model = YOLO(str(model_path), task="detect")
        metrics = model.val(
            data=args.data,
            split=args.split,
            imgsz=640,
            batch=1,
            device=args.device,
            workers=0,
            half=use_half,
            plots=False,
            channels=4,
            use_simotm="RGBT",
            pairs_rgb_ir=["visible", "infrared"],
            project=str(output),
            name=label.lower().replace(" ", "_"),
            exist_ok=True,
        )

        inference_ms = float(metrics.speed["inference"])
        results = metrics.results_dict
        row = {
            "model": label,
            "file": filename,
            "size_mb": round(model_path.stat().st_size / 1_000_000, 3),
            "precision": float(results["metrics/precision(B)"]),
            "recall": float(results["metrics/recall(B)"]),
            "map50": float(results["metrics/mAP50(B)"]),
            "map50_95": float(results["metrics/mAP50-95(B)"]),
            "preprocess_ms": float(metrics.speed["preprocess"]),
            "inference_ms": inference_ms,
            "postprocess_ms": float(metrics.speed["postprocess"]),
            "fps": 1000.0 / inference_ms,
        }
        rows.append(row)

    json_path = output / "comparison.json"
    csv_path = output / "comparison.csv"

    json_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()