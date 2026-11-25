# KAIST-OTS RGBT 模型部署

這個專案提供 KAIST → OTS 的 YOLO11 RGBT 模型、推論程式與驗證結果。

模型輸入：

- 可見光 RGB：3 通道
- 紅外線 IR：1 通道
- 合計：4 通道
- 輸入尺寸：640 × 640
- 類別：`human`、`dummy`

## 模型檔案

- `models/kaist_ots_best.pt`：原始 PyTorch 模型
- `models/kaist_ots.onnx`：ONNX 模型
- `models/kaist_ots_fp16.engine`：TensorRT FP16
- `models/kaist_ots_int8_mixed.engine`：TensorRT INT8/FP16 混合精度

## 測試環境

- Windows
- Python 3.9.13
- RTX 5070 Laptop GPU
- PyTorch 2.7.1 + CUDA 12.8
- TensorRT 10.14.1
- Batch size：1
- 驗證資料：743 張、1910 個標註

## 安裝

建立虛擬環境：

```powershell
py -3.9 -m venv .venv
```

安裝套件：

```powershell
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

## EO／IR 圖片命名

可見光圖片必須以 `_EO` 結尾，紅外線圖片必須以 `_IR` 結尾。

例如：

```text
PcsPhoto-001_EO.png
PcsPhoto-001_IR.png
```

兩張圖片需放在同一資料夾。

## 執行推論

```powershell
& ".\.venv\Scripts\python.exe" ".\scripts\inference_rgbt.py" `
  --model ".\models\kaist_ots_fp16.engine" `
  --source ".\samples\input" `
  --output ".\samples\output" `
  --name "fp16"
```

推論結果會包含：

- 畫好偵測框的 EO／IR 圖片
- `detections.json`
- 類別、信心值、座標與推論時間

## 重現驗證結果

先複製資料集設定：

```powershell
Copy-Item ".\configs\dataset.example.yaml" ".\configs\dataset.local.yaml"
```

修改 `configs/dataset.local.yaml`，填入本機資料集路徑。

執行四種格式的驗證：

```powershell
& ".\.venv\Scripts\python.exe" ".\scripts\validate_models.py" `
  --data ".\configs\dataset.local.yaml" `
  --split "val"
```

## Val 結果

| 模型 | 大小 MB | mAP50 | mAP50-95 | 推論 ms | FPS |
|---|---:|---:|---:|---:|---:|
| ORIGINAL FP16 | 7.95 | 0.9190 | 0.5433 | 13.36 | 74.85 |
| ONNX | 14.70 | 0.9188 | 0.5453 | 8.88 | 112.56 |
| TensorRT FP16 | 12.00 | 0.9179 | 0.5459 | 4.33 | 230.77 |
| TensorRT INT8/FP16 Mixed | 7.10 | 0.8713 | 0.5032 | 4.67 | 214.27 |

INT8 版本不是全層 INT8。輸入層、融合後 neck 及 Detect head 保留 FP16，中段 backbone 使用 INT8。

與 TensorRT FP16 相比：

- mAP50 下降 4.66 個百分點
- mAP50-95 下降 4.27 個百分點
- 模型大小從約 12.0 MB 降至約 7.1 MB

## TensorRT 注意事項

TensorRT engine 與顯卡、Windows 版本及 TensorRT 版本有關。

如果其他電腦無法載入 engine，請使用 `kaist_ots_best.pt` 或 `kaist_ots.onnx` 重新建立。

## 雲端備份

雲端下載連結：

```text
https://drive.google.com/file/d/1Ysd8KrbFML1RQ_6MWk0Wg5kvme9GCXjr/view?usp=sharing
```

## 授權

自訂 Ultralytics fork 的授權內容請參考 `LICENSE`。
## 報告用推論範例

`samples/report_inference_6sets` 收錄 6 組具有代表性的 VAL 場景。

挑選內容涵蓋：

- 0～5 個 human
- 每組包含 1 個 dummy
- 每組都有配對的 EO 可見光與 IR 熱影像
- TensorRT FP16 推論結果
- TensorRT INT8／FP16 Mixed Precision 推論結果
- `detections.json` 類別、信心分數、邊界框及速度紀錄

資料夾結構：

```text
samples/report_inference_6sets/
├─ input/          原始 EO／IR 影像
├─ fp16/           TensorRT FP16 推論結果
├─ int8_mixed/     INT8／FP16 混合精度推論結果
├─ README.txt
└─ selection.csv
這些圖片用於報告中的定性結果展示；正式 mAP 與速度數據仍使用完整的 743 張 VAL 資料集。
