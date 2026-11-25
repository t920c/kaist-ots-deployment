from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def find_pairs(source: Path):
    if source.is_file():
        eo_files = [source]
    else:
        eo_files = sorted(
            p for p in source.iterdir()
            if p.is_file() and p.stem.endswith("_EO")
        )

    pairs = []
    for eo in eo_files:
        if not eo.stem.endswith("_EO"):
            raise ValueError(f"Visible image must end with _EO: {eo}")

        ir = eo.with_name(eo.stem[:-3] + "_IR" + eo.suffix)
        if not ir.is_file():
            raise FileNotFoundError(f"Missing paired IR image: {ir}")

        pairs.append((eo, ir))

    if not pairs:
        raise FileNotFoundError(f"No *_EO images found in: {source}")

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="models/kaist_ots_fp16.engine",
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="runs/inference")
    parser.add_argument("--name", default="fp16")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    pairs = find_pairs(Path(args.source))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model, task="detect")
    summary = []

    for eo, ir in pairs:
        result = model.predict(
            source=str(eo),
            imgsz=640,
            conf=args.conf,
            device=args.device,
            save=True,
            project=str(output),
            name=args.name,
            exist_ok=True,
            channels=4,
            use_simotm="RGBT",
            pairs_rgb_ir=["_EO", "_IR"],
            verbose=False,
        )[0]

        detections = []
        for box in result.boxes:
            detections.append({
                "class_id": int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 6),
                "xyxy": [round(float(x), 2) for x in box.xyxy[0]],
            })

        summary.append({
            "eo": str(eo),
            "ir": str(ir),
            "detections": detections,
            "speed_ms": result.speed,
        })

        print(f"{eo.name}: {len(detections)} detections")

    report = output / args.name / "detections.json"
    report.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Results saved to: {output / args.name}")


if __name__ == "__main__":
    main()